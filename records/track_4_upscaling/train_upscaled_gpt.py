from __future__ import annotations

import csv
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.func import functional_call

from .model import GPT, GPTConfig, non_embedding_params, init_params
from .training import Hyperparameters, train_model, append_csv, TrainResult


# ---------------------------------------------------------------------------
# Growth methods: stack_pad (function-preserving stacking + zero padding) and LiGO operators
# ---------------------------------------------------------------------------

_BLOCK_RESTS = [
    "attn.q.weight", "attn.q.bias",
    "attn.k.weight", "attn.k.bias",
    "attn.v.weight", "attn.v.bias",
    "attn.proj.weight", "attn.proj.bias",
    "mlp.fc.weight", "mlp.fc.bias",
    "mlp.proj.weight", "mlp.proj.bias",
    "norm1.gains", "norm2.gains",
]


def _identity_expand(d_out: int, d_in: int, generator: torch.Generator) -> Tensor:
    """Net2Net-style expansion: identity on the shared block, new rows duplicate
    a random existing unit (a sane init that fitting then refines)."""
    R = torch.zeros(d_out, d_in)
    m = min(d_out, d_in)
    R[:m, :m] = torch.eye(m)
    for j in range(d_in, d_out):
        r = int(torch.randint(0, d_in, (1,), generator=generator))
        R[j, r] = 1.0
    return R


def _dus_select(L2: int, L: int) -> Tensor:
    """DUS-style depth init: target layer i copies small layer (i % L)."""
    C = torch.zeros(L2, L)
    for i in range(L2):
        C[i, i % L] = 1.0
    return C


class LigoOperators(nn.Module):
    """Width operators (Rd for the residual/attn axis, Rm for the MLP hidden
    axis) and a depth operator C. Only the dims that actually change are
    learnable; unchanged axes are fixed identities."""

    def __init__(self, small: GPTConfig, target: GPTConfig, seed: int = 0):
        super().__init__()
        assert small.head_dim == target.head_dim, "LiGO keeps head_dim fixed (grows num_heads)"
        g = torch.Generator().manual_seed(seed)
        d, d2 = small.model_dim, target.model_dim
        m, m2 = small.mlp_ratio * d, target.mlp_ratio * d2
        L, L2 = small.num_layers, target.num_layers
        self.width_changes = d2 != d
        self.depth_changes = L2 != L

        Rd = _identity_expand(d2, d, g)
        Rm = _identity_expand(m2, m, g)
        C = _dus_select(L2, L)
        if self.width_changes:
            self.Rd = nn.Parameter(Rd)
            self.Rm = nn.Parameter(Rm)
        else:
            self.register_buffer("Rd", Rd)
            self.register_buffer("Rm", Rm)
        if self.depth_changes:
            self.C = nn.Parameter(C)
        else:
            self.register_buffer("C", C)


def _expand_block(small_params: dict, j: int, Rd: Tensor, Rm: Tensor) -> dict:
    """Width-expand small block j's tensors (no-op if Rd, Rm are identity)."""
    def b(rest):
        return small_params[f"blocks.{j}.{rest}"]
    out = {}
    for rest in ("attn.q.weight", "attn.k.weight", "attn.v.weight", "attn.proj.weight"):
        out[rest] = Rd @ b(rest) @ Rd.T
    for rest in ("attn.q.bias", "attn.k.bias", "attn.v.bias", "attn.proj.bias"):
        out[rest] = Rd @ b(rest)
    out["mlp.fc.weight"] = Rm @ b("mlp.fc.weight") @ Rd.T
    out["mlp.fc.bias"] = Rm @ b("mlp.fc.bias")
    out["mlp.proj.weight"] = Rd @ b("mlp.proj.weight") @ Rm.T
    out["mlp.proj.bias"] = Rd @ b("mlp.proj.bias")
    out["norm1.gains"] = Rd @ b("norm1.gains")
    out["norm2.gains"] = Rd @ b("norm2.gains")
    return out


def materialize(small_params: dict, ops: LigoOperators,
                small: GPTConfig, target: GPTConfig) -> dict:
    """Build the full target parameter dict as a differentiable function of the
    (frozen) small params and the LiGO operators."""
    Rd, Rm, C = ops.Rd, ops.Rm, ops.C
    L, L2 = small.num_layers, target.num_layers

    expanded = [_expand_block(small_params, j, Rd, Rm) for j in range(L)]

    params: dict = {}
    params["embed.weight"] = small_params["embed.weight"] @ Rd.T
    params["proj.weight"] = small_params["proj.weight"] @ Rd.T
    params["proj.bias"] = small_params["proj.bias"]
    params["norm1.gains"] = Rd @ small_params["norm1.gains"]
    params["norm2.gains"] = Rd @ small_params["norm2.gains"]
    for i in range(L2):
        for rest in _BLOCK_RESTS:
            acc = None
            for j in range(L):
                term = C[i, j] * expanded[j][rest]
                acc = term if acc is None else acc + term
            params[f"blocks.{i}.{rest}"] = acc
    return params


@torch.no_grad()
def _eval_functional(target_model: GPT, params: dict, batch) -> float:
    inputs, targets = batch
    loss = functional_call(target_model, params, (inputs, targets))
    return float(loss) / targets.numel()


def ligo_grow(small_model: GPT, target_config: GPTConfig, train_gen, eval_batch,
              *, fit_steps: int = 100, lr: float = 3e-4, tokens_per_step: int = 0,
              device: torch.device = torch.device("cpu"), seed: int = 0,
              grad_sync=None) -> dict:
    """Grow ``small_model`` to ``target_config`` via fitted LiGO operators.

    Under torch.distributed, ``grad_sync`` averages the operator gradients across
    ranks each fit step; combined with the identical seeded operator init this makes
    every rank converge to the same operators, so the grown model is rank-consistent.

    Returns dict(state, config, fit_tokens, fit_flops, naive_loss, post_loss).
    """
    small_config = small_model.config
    small_params = {n: p.detach().to(device) for n, p in small_model.named_parameters()}
    ops = LigoOperators(small_config, target_config, seed=seed).to(device)
    target_model = GPT(target_config).to(device)
    target_model.train()

    naive_loss = _eval_functional(target_model, materialize(small_params, ops, small_config, target_config), eval_batch)

    trainable = [p for p in ops.parameters() if p.requires_grad]
    post_loss = naive_loss
    fit_flops = 0.0
    if trainable and fit_steps > 0:
        from torch.utils.flop_counter import FlopCounterMode
        opt = torch.optim.AdamW(trainable, lr=lr)

        def fit_step():
            inputs, targets = next(train_gen)
            params = materialize(small_params, ops, small_config, target_config)
            loss = functional_call(target_model, params, (inputs, targets)) / targets.numel()
            opt.zero_grad()
            loss.backward()
            if grad_sync is not None:
                grad_sync(ops)
            opt.step()

        # measure one fit step (shapes are fixed) and charge it across all steps and
        # ranks, matching the global FLOP convention used for training steps
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        counter = FlopCounterMode(display=False)
        with counter:
            fit_step()
        fit_flops = float(counter.get_total_flops()) * fit_steps * world_size
        for _ in range(fit_steps - 1):
            fit_step()
        post_loss = _eval_functional(target_model, materialize(small_params, ops, small_config, target_config), eval_batch)

    final = materialize(small_params, ops, small_config, target_config)
    state = {k: v.detach().clone() for k, v in final.items()}
    return {
        "state": state,
        "config": target_config,
        "fit_tokens": fit_steps * tokens_per_step,
        "fit_flops": fit_flops,
        "naive_loss": naive_loss,
        "post_loss": post_loss,
    }


def load_grown(state: dict, config: GPTConfig, device: torch.device = torch.device("cpu")) -> GPT:
    """Instantiate a GPT and load grown params (buffers come from the fresh model)."""
    model = GPT(config).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.endswith("angular_freq")]
    assert not bad, f"missing params when loading grown model: {bad}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    return model


# ---------------------------------------------------------------------------
# stack_pad: function-preserving width+depth growth (DUS generalized)
# ---------------------------------------------------------------------------

def _pad2d(w: Tensor, out2: int, in2: int) -> Tensor:
    o = torch.zeros(out2, in2, dtype=w.dtype, device=w.device)
    o[:w.size(0), :w.size(1)] = w
    return o


def _pad1d(v: Tensor, out2: int) -> Tensor:
    o = torch.zeros(out2, dtype=v.dtype, device=v.device)
    o[:v.size(0)] = v
    return o


def _interleave_positions(L: int, L2: int) -> list:
    """Map each target block to a source small block, spreading the (L2-L) new
    identity blocks evenly. Returns a length-L2 list with ints (source block) or
    None (new identity block)."""
    n_new = L2 - L
    if n_new <= 0:
        return list(range(L2))
    step = L2 / n_new
    new_slots = {int((k + 0.5) * step) for k in range(n_new)}
    src, j = [], 0
    for i in range(L2):
        if i in new_slots:
            src.append(None)
        else:
            src.append(j)
            j += 1
    return src


def stack_pad_grow(small_model: GPT, target: GPTConfig, device: torch.device) -> dict:
    """Function-preserving "stack + pad" init.

    Copies the small weights into the top-left sub-blocks of the bigger tensors,
    zeros the new output dims (extra width channels, extra attention heads,
    extra MLP units) and the new layers' residual projections so they are inert,
    and rescales the RMSNorm gains by sqrt(d2/d) to cancel the change in the
    normalized dimension. The grown model then computes exactly the small model's
    function at init (naive loss == L1), with no fit phase. New width channels
    stay zero throughout the network, so the head reads only the original ones.
    """
    sp = {n: p.detach().to(device) for n, p in small_model.named_parameters()}
    s = small_model.config
    d, d2 = s.model_dim, target.model_dim
    h, h2 = s.mlp_ratio * d, target.mlp_ratio * d2
    L, L2 = s.num_layers, target.num_layers
    assert d2 >= d and h2 >= h and L2 >= L, "stack_pad transfer grows up only"
    # RMSNorm averages over d2 dims (the extra d2-d channels are zero), so the
    # kept channels' normalized values grow by sqrt(d2/d); scale gains down to
    # cancel it and keep the function identical.
    norm_scale = (d / d2) ** 0.5

    base = GPT(target).to(device)
    init_params(base)  # fresh blocks are already identities (proj weights zero-init)
    state = {k: v.detach().clone() for k, v in base.state_dict().items()
             if not k.endswith("angular_freq")}

    def scaled_gains(name: str) -> Tensor:
        g = _pad1d(sp[name], d2)
        g[:d] = sp[name] * norm_scale
        return g

    state["embed.weight"] = _pad2d(sp["embed.weight"], target.vocab_size, d2)
    state["norm1.gains"] = scaled_gains("norm1.gains")
    state["norm2.gains"] = scaled_gains("norm2.gains")
    state["proj.weight"] = _pad2d(sp["proj.weight"], target.vocab_size, d2)
    state["proj.bias"] = sp["proj.bias"].clone()

    for i, j in enumerate(_interleave_positions(L, L2)):
        if j is None:
            continue  # keep the fresh identity block
        tp, sq = f"blocks.{i}.", f"blocks.{j}."
        for nm in ("attn.q", "attn.k", "attn.v", "attn.proj"):
            state[tp + nm + ".weight"] = _pad2d(sp[sq + nm + ".weight"], d2, d2)
            state[tp + nm + ".bias"] = _pad1d(sp[sq + nm + ".bias"], d2)
        state[tp + "mlp.fc.weight"] = _pad2d(sp[sq + "mlp.fc.weight"], h2, d2)
        state[tp + "mlp.fc.bias"] = _pad1d(sp[sq + "mlp.fc.bias"], h2)
        state[tp + "mlp.proj.weight"] = _pad2d(sp[sq + "mlp.proj.weight"], d2, h2)
        state[tp + "mlp.proj.bias"] = _pad1d(sp[sq + "mlp.proj.bias"], d2)
        state[tp + "norm1.gains"] = scaled_gains(sq + "norm1.gains")
        state[tp + "norm2.gains"] = scaled_gains(sq + "norm2.gains")

    return {"state": state, "config": target}


# ---------------------------------------------------------------------------
# Upscaling: load the small model -> grow -> resume training to L2, measure cost
# ---------------------------------------------------------------------------

RECORD_FIELDS = [
    "date", "axis", "op", "small_params", "target_params", "L1", "L2",
    "hparams_version", "fit_steps",
    "steps_to_L2", "tokens_to_L2", "flops_warmstart", "wall_seconds", "reached",
    "n_seeds", "p_value", "contributor",
    # periodic (filled when references are (re)run):
    "ref_version", "c_amortized", "c_cold",
]


def read_references(path: str) -> list[dict]:
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def _find_ref(refs: list[dict], kind: str, ref_version: str, target_params: int):
    """Best match by kind + ref_version, preferring the closest param count."""
    cands = [r for r in refs if r["kind"] == kind and r["ref_version"] == ref_version]
    if not cands:
        return None
    return min(cands, key=lambda r: abs(int(r["params_nonembed"]) - target_params))


@dataclass
class UpscaleResult:
    record: dict
    grown_model: GPT
    resume: TrainResult


def grow_model(small_model: GPT, op: str, target_config: GPTConfig, *,
               train_gen=None, eval_batch=None, fit_steps: int = 100, tokens_per_step: int = 0,
               device=torch.device("cpu"), seed: int = 0, grad_sync=None) -> dict:
    """Grow ``small_model`` into the bigger ``target_config``.

    Returns dict(model, config, fit_tokens, naive_loss?, post_loss?).

    op="stack_pad": function-preserving "stack + pad" init for width+depth growth
               (copy into sub-blocks, zero new outputs, rescale norm gains); no fit.
               Generalizes DUS (depth-only) to simultaneous width+depth growth.
    op="ligo": learned linear operators; handles width and/or depth growth.
    """
    if op == "stack_pad":
        out = stack_pad_grow(small_model, target_config, device)
        model = load_grown(out["state"], target_config, device)
        naive = None
        if eval_batch is not None:
            with torch.no_grad():
                naive = float(model(*eval_batch)) / eval_batch[1].numel()
        return {"model": model, "config": target_config, "fit_tokens": 0, "fit_flops": 0.0,
                "naive_loss": naive, "post_loss": naive}
    elif op == "ligo":
        out = ligo_grow(small_model, target_config, train_gen, eval_batch,
                        fit_steps=fit_steps, tokens_per_step=tokens_per_step,
                        device=device, seed=seed, grad_sync=grad_sync)
        model = load_grown(out["state"], target_config, device)
        return {"model": model, "config": target_config, "fit_tokens": out["fit_tokens"],
                "fit_flops": out["fit_flops"],
                "naive_loss": out["naive_loss"], "post_loss": out["post_loss"]}
    raise ValueError(f"unknown op {op!r}")


def run_upscale(
    small_model: GPT,
    *,
    op: str,
    target_config: GPTConfig,
    hparams: Hyperparameters,
    L1: float,
    L2: float,
    train_gen,
    val_batches,
    fit_steps: int = 100,
    max_steps: int = 400,
    schedule_steps: int | None = None,
    eval_every: int = 25,
    references_path: str | None = None,
    ref_version: str = "ref1",
    small_flops: float = 0.0,
    records_path: str | None = None,
    axis: str = "transfer",
    contributor: str = "smoke",
    device=torch.device("cpu"),
    seed: int = 0,
    log: bool = False,
    amp_dtype=None,
    grad_sync=None,
    flops_per_step: float,
    world_size: int = 1,
    compile_fn=lambda m: m,
) -> UpscaleResult:
    small_params = non_embedding_params(small_model)
    tokens_per_step = hparams.batch_seqs * hparams.seq_len * world_size  # global tokens/step

    grown = grow_model(small_model, op, target_config,
                       train_gen=train_gen, eval_batch=val_batches[0],
                       fit_steps=fit_steps, tokens_per_step=tokens_per_step,
                       device=device, seed=seed, grad_sync=grad_sync)
    model = grown["model"]
    target_params = non_embedding_params(model)
    op_label = op
    model = compile_fn(model)  # compile the bigger model for the resume training

    resume = train_model(model, train_gen, val_batches, hparams=hparams,
                         max_steps=max_steps, schedule_steps=schedule_steps,
                         target_loss=L2, eval_every=eval_every,
                         log=log, amp_dtype=amp_dtype, grad_sync=grad_sync,
                         flops_per_step=flops_per_step, world_size=world_size)

    warmstart_tokens = resume.tokens + grown["fit_tokens"]
    flops_warmstart = flops_per_step * resume.steps + grown["fit_flops"]

    record = {
        "date": "", "axis": axis, "op": op_label,
        "small_params": small_params, "target_params": target_params,
        "L1": L1, "L2": L2, "hparams_version": hparams.version,
        "fit_steps": fit_steps if op == "ligo" else 0,
        "steps_to_L2": resume.steps, "tokens_to_L2": warmstart_tokens,
        "flops_warmstart": flops_warmstart, "wall_seconds": round(resume.wall_seconds, 1),
        "reached": int(resume.reached_target),
        "n_seeds": 1, "p_value": "", "contributor": contributor,
        "ref_version": "", "c_amortized": "", "c_cold": "",
        # not persisted (not in RECORD_FIELDS); handy for diagnostics/asserts:
        "ligo_naive_loss": grown.get("naive_loss"),
        "ligo_post_loss": grown.get("post_loss"),
    }

    # periodic c vs the from-scratch-large reference
    if references_path:
        refs = read_references(references_path)
        fs = _find_ref(refs, "from_scratch_large", ref_version, target_params)
        if fs:
            denom = float(fs["flops_to_L2"])
            record["ref_version"] = ref_version
            record["c_amortized"] = flops_warmstart / denom
            record["c_cold"] = (flops_warmstart + small_flops) / denom

    if records_path:
        append_csv(records_path, record, RECORD_FIELDS)

    return UpscaleResult(record=record, grown_model=model, resume=resume)
