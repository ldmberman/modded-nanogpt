"""Model-coupled half of Track 5 circuit discovery.

Implements the `tap(name, t)` protocol that `GPT._cd_tap` calls, plus the three
passes that drive the model-independent search in `circuit.py`:

1. mean pass  -> per-node mean activation over the task examples (ablation baseline)
2. clean pass -> clean activations + node attribution scores (one backward; nodes
                 retain grad so each score is (clean - mean) . dM/dnode, the
                 first-order mean-ablation effect on the metric M)
3. patch pass -> `measure(kept)`: forward with every node outside `kept` frozen to
                 its mean activation; returns mean metric M (logit diff).

Node set (v1): component-level writers exposed by the fused forward --
`attn{i}` (i != 6), `mlp{i}`, and `skip6`. The `embed` tap is the autograd root
(not ablatable); `resid_final` is the readout (not ablatable). Per-head edges
need an eager attention reimplementation (FA3 hides internals) -- a later upgrade.

Metric: M = logit(correct) - logit(distractor) at the answer position, using raw
`lm_head` logits (monotonic with the soft-capped decision, cleaner gradients).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .circuit import (CDExample, CircuitResult, geomean, greedy_minimal_circuit,
                      random_faithfulness_at_size)

EOT_ID = 50256


def ablatable_nodes(num_layers: int = 11) -> list[str]:
    nodes = [f"attn{i}" for i in range(num_layers) if i != 6]
    nodes += [f"mlp{i}" for i in range(num_layers)]
    nodes += ["skip6"]
    return nodes


class Recorder:
    """Stateful tap target. `GPT._cd` is set to an instance; `mode` selects pass."""

    READOUT = "resid_final"
    ROOT = "embed"

    def __init__(self):
        self.mode: str | None = None
        self._sum: dict[str, Tensor] = {}
        self._cnt: dict[str, int] = {}
        self.means: dict[str, Tensor] = {}
        self.clean: dict[str, Tensor] = {}
        self.readout: Tensor | None = None
        self.kept: frozenset[str] | None = None

    def reset_pass(self):
        self.clean = {}
        self.readout = None

    def tap(self, name: str, t: Tensor) -> Tensor:
        if self.mode == "mean":
            if name in (self.READOUT, self.ROOT):
                return t
            flat = t.detach().reshape(-1, t.shape[-1]).float()
            if name in self._sum:
                self._sum[name] += flat.sum(0)
                self._cnt[name] += flat.shape[0]
            else:
                self._sum[name] = flat.sum(0)
                self._cnt[name] = flat.shape[0]
            return t

        if self.mode == "clean":
            if name == self.READOUT:
                self.readout = t  # stays on graph for the metric + backward
                return t
            if name == self.ROOT:
                leaf = t.detach().requires_grad_(True)  # autograd root
                self.clean[name] = leaf
                return leaf
            t.retain_grad()  # already requires grad via the root path
            self.clean[name] = t
            return t

        if self.mode == "patch":
            if name == self.READOUT:
                self.readout = t
                return t
            if name == self.ROOT:
                return t
            if self.kept is not None and name not in self.kept:
                m = self.means[name].to(t.dtype)
                return m.view(*([1] * (t.ndim - 1)), -1).expand_as(t)
            return t

        return t

    def finalize_means(self):
        self.means = {k: self._sum[k] / self._cnt[k] for k in self._sum}


class TaskPatcher:
    """Runs the three passes for one task's CDExamples on a real GPT model."""

    def __init__(self, model, schedule_cfg, get_bigram_hash, examples: list[CDExample],
                 device: str = "cuda", nodes: list[str] | None = None):
        self.model = model
        self.cfg = schedule_cfg
        self.get_bigram_hash = get_bigram_hash
        self.examples = examples
        self.device = device
        self.nodes = nodes if nodes is not None else ablatable_nodes(model.num_layers)
        self.rec = Recorder()
        self._inputs = [self._prep(ex) for ex in examples]
        self._measure_cache: dict[frozenset, float] = {}

    def _prep(self, ex: CDExample):
        ids = torch.tensor(ex.context, dtype=torch.int32)
        T0 = ids.numel()
        # FA3/FP8 require the packed length to be a multiple of 16. Right-pad with
        # EOT; trailing tokens sit after the (causal) readout position so they
        # cannot affect the logits we read at read_pos.
        T = ((T0 + 15) // 16) * 16
        if T > T0:
            ids = torch.cat([ids, torch.full((T - T0,), EOT_ID, dtype=torch.int32)])
        cum = torch.full((128,), T, dtype=torch.int32)
        cum[0] = 0
        cum[1] = T
        target = torch.zeros(T, dtype=torch.int64)
        bigram = self.get_bigram_hash(ids)
        return dict(input_seq=ids.to(self.device), target=target.to(self.device),
                    seqlens=cum.to(self.device), bigram=bigram.to(self.device),
                    read_pos=ex.read_pos, correct=ex.correct_id, distractor=ex.distractor_id)

    def _forward(self, inp):
        self.model(input_seq=inp["input_seq"], target_seq=inp["target"], seqlens=inp["seqlens"],
                   bigram_input_seq=inp["bigram"], schedule_cfg=self.cfg)

    def _metric_from_readout(self, inp) -> Tensor:
        x = self.rec.readout[0, inp["read_pos"]]            # [D]
        # lm_head uses transposed storage [dim, vocab]: logits = x @ W (pre-softcap)
        w = self.model.lm_head.weight
        logits = x @ w.type_as(x)                           # [vocab]
        return logits[inp["correct"]] - logits[inp["distractor"]]

    @torch.no_grad()
    def compute_means(self):
        self.rec.mode = "mean"
        self.rec._sum = {}
        self.rec._cnt = {}
        for inp in self._inputs:
            self._forward(inp)
        self.rec.finalize_means()

    def compute_scores(self) -> dict[str, float]:
        """One backward per example; node score = sum_examples (clean-mean).grad."""
        scores = {n: 0.0 for n in self.nodes}
        self.rec.mode = "clean"
        for inp in self._inputs:
            self.rec.reset_pass()
            with torch.enable_grad():
                self._forward(inp)
                m = self._metric_from_readout(inp)
            m.backward()
            for n in self.nodes:
                t = self.rec.clean.get(n)
                if t is None or t.grad is None:
                    continue
                contrib = ((t.detach() - self.means[n].to(t.dtype)) * t.grad).sum().item()
                scores[n] += contrib
        return scores

    @property
    def means(self):
        return self.rec.means

    @torch.no_grad()
    def measure(self, kept: frozenset[str]) -> float:
        if kept in self._measure_cache:
            return self._measure_cache[kept]
        self.rec.mode = "patch"
        self.rec.kept = kept
        total = 0.0
        for inp in self._inputs:
            self.rec.reset_pass()
            self._forward(inp)
            total += self._metric_from_readout(inp).item()
        val = total / len(self._inputs)
        self._measure_cache[kept] = val
        return val


def discover_task_circuit(model, schedule_cfg, get_bigram_hash, examples: list[CDExample],
                          tau: float = 0.8, device: str = "cuda",
                          random_trials: int = 20) -> dict:
    tp = TaskPatcher(model, schedule_cfg, get_bigram_hash, examples, device=device)
    model.requires_grad_(False)  # grads flow to activation leaves, not params
    model._cd_enabled = True
    model._cd = tp.rec
    try:
        tp.compute_means()
        scores = tp.compute_scores()
        res: CircuitResult = greedy_minimal_circuit(
            tp.nodes, scores, tp.measure, tau=tau, task=examples[0].task)
        rand_mean, rand_std = random_faithfulness_at_size(
            tp.nodes, tp.measure, size=res.size, trials=random_trials)
    finally:
        model._cd_enabled = False
        model._cd = None
    return dict(result=res, scores=scores, rand_mean=rand_mean, rand_std=rand_std)


def run(model, schedule_cfg, get_bigram_hash, print0, tasks: list[str],
        n_cd: int = 48, tau: float = 0.8, seed: int = 0, device: str = "cuda"):
    from .probe_tasks import build_all
    from .circuit import build_cd_inputs

    model.eval()
    built = build_all(n_per_task=n_cd, seed=seed, tasks=tasks)
    rows = []
    for task in tasks:
        cds = build_cd_inputs(built[task])
        out = discover_task_circuit(model, schedule_cfg, get_bigram_hash, cds,
                                    tau=tau, device=device)
        r = out["result"]
        rows.append((task, r))
        print0(f"[cd] {task:22} size={r.size:3d}  F={r.faithfulness:.3f}  "
               f"M_full={r.m_full:+.3f} M_empty={r.m_empty:+.3f}  "
               f"rand@{r.size}={out['rand_mean']:.3f}±{out['rand_std']:.3f}", console=True)

    sizes = [r.size for _, r in rows if r.size > 0]
    print0("=" * 72, console=True)
    print0(f"Track 5 circuit discovery  (tau={tau}, n_cd={n_cd})", console=True)
    print0(f"geomean circuit size = {geomean(sizes):.2f} over {len(sizes)} tasks", console=True)
    print0("=" * 72, console=True)
    return rows
