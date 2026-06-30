from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 64
    num_layers: int = 2
    model_dim: int = 64
    head_dim: int = 16
    mlp_ratio: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GPTConfig":
        return GPTConfig(**{k: d[k] for k in GPTConfig().to_dict() if k in d})


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE; dim must be divisible by 4
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor) -> Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 16):
        super().__init__()
        assert dim % head_dim == 0, f"model_dim {dim} not divisible by head_dim {head_dim}"
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for rotary"
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)
        self.scale = head_dim ** -0.5

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK-norm
        q, k = self.rotary(q), self.rotary(k)
        # QK-norm/rotary run in fp32; cast back to v's dtype so SDPA uses the
        # memory-light flash kernel under autocast (fp32 inputs force the math
        # path, which materializes the full B*H*T*T scores and blows up memory).
        q, k = q.to(v.dtype), k.to(v.dtype)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            scale=self.scale, is_causal=True,
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hdim = mlp_ratio * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.fc(x).relu().square())  # ReLU^2


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.model_dim).to(dtype)
        self.blocks = nn.ModuleList(
            [Block(config.model_dim, config.head_dim, config.mlp_ratio) for _ in range(config.num_layers)]
        )
        self.proj = Linear(config.model_dim, config.vocab_size)
        self.norm1 = RMSNorm(config.model_dim)
        self.norm2 = RMSNorm(config.model_dim)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        x = self.norm2(x).reshape(targets.numel(), -1)
        t = targets.reshape(-1)
        # Chunked LM head + loss. Materializing the full (B*T, vocab) fp32 logits
        # (and the softcap's temporaries) at once dominates activation memory at
        # this batch/vocab; chunking bounds it. The result is exact (sum of
        # per-chunk CE sums == CE sum over all rows).
        rows_per_chunk = 4096
        n_chunks = max(1, (x.size(0) + rows_per_chunk - 1) // rows_per_chunk)
        total = x.new_zeros((), dtype=torch.float32)
        for xc, tc in zip(x.chunk(n_chunks), t.chunk(n_chunks)):
            logits = self.proj(xc).float()
            logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()  # softcap-ish
            total = total + F.cross_entropy(logits, tc, reduction="sum")
        return total


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_params(model: GPT) -> None:
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()
            else:
                w.normal_(std=(0.33 ** 0.5) / w.size(-1) ** 0.5)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.fill_(1.0)
        else:
            raise RuntimeError(f"Uninitialized parameter: {name}")


# ---------------------------------------------------------------------------
# Optimizers (Muon on blocks, AdamW on embed/head/scalars)
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    work_dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(work_dtype)
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.numel(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        initialized = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if initialized else 1
        rank = dist.get_rank() if initialized else 0
        for group in self.param_groups:
            params = group["params"]
            # Each param is owned by one rank (round-robin over the numel-sorted list,
            # so work is balanced). The owner computes the update; the result is then
            # broadcast. We use per-param broadcast rather than a round-robin all_gather
            # because Muon's matrices have heterogeneous shapes and the distributed
            # backends require a uniform shape across an all_gather list.
            for i, p in enumerate(params):
                owner = i % world_size
                if owner == rank:
                    state = self.state[p]
                    if not state:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                if world_size > 1:
                    dist.broadcast(p, src=owner)


def build_optimizers(model: GPT, *, embed_lr=0.3, head_lr=0.004, scalar_lr=0.015,
                     muon_lr=0.025, muon_wd=0.05, betas=(0.8, 0.95)):
    """Return [AdamW(embed/head/scalars), Muon(block matrices)]."""
    fused = torch.cuda.is_available()
    adam = torch.optim.AdamW(
        [dict(params=[model.embed.weight], lr=embed_lr),
         dict(params=[model.proj.weight], lr=head_lr),
         dict(params=[p for p in model.parameters() if p.ndim < 2], lr=scalar_lr)],
        betas=betas, eps=1e-10, weight_decay=0.001, fused=fused,
    )
    muon_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
    muon = Muon(muon_params, lr=muon_lr, weight_decay=muon_wd)
    # sanity: every parameter is optimized exactly once
    covered = {id(p) for opt in (adam, muon) for g in opt.param_groups for p in g["params"]}
    missing = [n for n, p in model.named_parameters() if id(p) not in covered]
    assert not missing, f"parameters not covered by optimizer: {missing}"
    return [adam, muon]


# ---------------------------------------------------------------------------
# Size / FLOPs helpers
# ---------------------------------------------------------------------------

def count_params(model: GPT, include_embedding: bool = True) -> int:
    total = 0
    for name, p in model.named_parameters():
        if not include_embedding and name == "embed.weight":
            continue
        total += p.numel()
    return total


def non_embedding_params(model: GPT) -> int:
    """Size metric: everything except the token embedding."""
    return count_params(model, include_embedding=False)


def measure_step_flops(model: GPT, batch, optimizers=None, *,
                       autocast_dtype: torch.dtype | None = None) -> int:
    """FLOPs actually executed in one training step: forward + backward, plus the
    optimizer update when ``optimizers`` is given.

    Uses PyTorch's dispatch-level counter, so it sees attention, embeddings, the
    head, and Muon's Newton-Schulz iterations, and it skips the gradient matmuls
    of frozen params. Shapes are fixed across steps, so the run total is this
    times the step count. ``optimizers`` mutate params, so pass a throwaway clone
    of the model if you need to measure before/without training it for real.
    """
    from torch.utils.flop_counter import FlopCounterMode
    inputs, targets = batch
    was_training = model.training
    model.train()
    counter = FlopCounterMode(display=False)
    with counter:
        if autocast_dtype is not None and inputs.is_cuda:
            with torch.autocast("cuda", dtype=autocast_dtype):
                loss = model(inputs, targets)
        else:
            loss = model(inputs, targets)
        (loss / targets.numel()).backward()
        if optimizers is not None:
            for opt in optimizers:
                opt.step()
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return int(counter.get_total_flops())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


GPT2_VOCAB_SIZE = 50304


def _load_data_shard(path) -> Tensor:
    from pathlib import Path
    path = Path(path)
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "token count does not match header"
    return tokens


def fineweb_data_generator(filename_pattern: str, batch_seqs: int, seq_len: int,
                           device: torch.device, *, rank: int = 0, world_size: int = 1):
    """Yield this rank's (inputs, targets) of shape [batch_seqs, seq_len] forever,
    streaming over all shards matching ``filename_pattern`` (relative to cwd).
    Each step consumes a global batch of ``world_size * batch_seqs`` sequences and
    hands rank ``rank`` its disjoint slice."""
    from pathlib import Path
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no shards match {filename_pattern!r} under {Path.cwd()}"
    local_tokens = batch_seqs * seq_len
    global_tokens = world_size * local_tokens
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + global_tokens + 1 >= len(tokens):
            try:
                tokens = _load_data_shard(next(file_iter))
            except StopIteration:
                file_iter = iter(files)
                tokens = _load_data_shard(next(file_iter))
            pos = 0
        start = pos + rank * local_tokens
        buf = tokens[start:start + local_tokens + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32)
        targets = buf[1:].to(device=device, dtype=torch.int64)
        pos += global_tokens
        yield inputs.view(batch_seqs, seq_len), targets.view(batch_seqs, seq_len)


def fineweb_val_batches(filename_pattern: str, batch_seqs: int, seq_len: int,
                        device: torch.device, *, total_tokens: int,
                        rank: int = 0, world_size: int = 1):
    """A fixed list of (inputs, targets) validation batches covering ~total_tokens
    across all ranks (a deterministic prefix of the val shards). Rank ``rank`` gets
    its disjoint slice each step, so callers reduce per-rank losses to a global mean."""
    from pathlib import Path
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no shards match {filename_pattern!r} under {Path.cwd()}"
    local_tokens = batch_seqs * seq_len
    global_tokens = world_size * local_tokens
    n_steps = max(1, total_tokens // global_tokens)
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    batches = []
    for _ in range(n_steps):
        if pos + global_tokens + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        start = pos + rank * local_tokens
        buf = tokens[start:start + local_tokens + 1]
        inputs = buf[:-1].to(device=device, dtype=torch.int32)
        targets = buf[1:].to(device=device, dtype=torch.int64)
        batches.append((inputs.view(batch_seqs, seq_len), targets.view(batch_seqs, seq_len)))
        pos += global_tokens
    return batches
