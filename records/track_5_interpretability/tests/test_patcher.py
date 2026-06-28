"""Offline integration test for the model-coupled Patcher (CPU, no FA3/GPU).

Uses a tiny autograd "model" that speaks the same tap protocol as `GPT`
(`_cd_tap`, `_cd_enabled`, `_cd`, `lm_head.weight`, `num_layers`) and a forward
matching the real signature. One ablatable node ("attn0") carries all the signal;
the other ("mlp0") contributes exactly zero. We assert the full pipeline recovers
the known minimal circuit {attn0}.
"""

from __future__ import annotations

import torch
from torch import nn

from records.track_5_interpretability import circuit_patcher as P
from records.track_5_interpretability.circuit import CDExample


class TinyModel(nn.Module):
    _cd_enabled = False

    def _cd_tap(self, name, t):
        if self._cd_enabled:
            return self._cd.tap(name, t)
        return t

    def __init__(self, vocab=16, dim=8, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.num_layers = 11
        self.embed_table = nn.Parameter(torch.randn(vocab, dim, generator=g))
        self.A0 = nn.Parameter(torch.randn(dim, dim, generator=g))
        self.A1 = nn.Parameter(torch.zeros(dim, dim))  # mlp0 contributes nothing
        head = nn.Module()  # transposed [dim, vocab] storage like CastedLinearT
        head.weight = nn.Parameter(torch.randn(dim, vocab, generator=g))
        self.lm_head = head

    def forward(self, input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg):
        x = self.embed_table[input_seq.long()].unsqueeze(0)   # [1,T,D]
        x = self._cd_tap("embed", x)
        a0 = torch.relu(x @ self.A0)
        a0 = self._cd_tap("attn0", a0)
        x = x + a0
        m0 = x @ self.A1
        m0 = self._cd_tap("mlp0", m0)
        x = x + m0
        x = self._cd_tap("resid_final", x)
        return x.sum().reshape(1)


def _examples(n=12, T=16, vocab=16, seed=0):  # T multiple of 16 -> no EOT padding
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        ctx = torch.randint(0, vocab, (T,), generator=g).tolist()
        c, d = 1, 2
        out.append(CDExample(context=ctx, read_pos=T - 1, correct_id=c,
                             distractor_id=d, task="synthetic"))
    return out


def test_pipeline_recovers_known_minimal_circuit():
    model = TinyModel()
    exs = _examples()
    out = P.discover_task_circuit(
        model, schedule_cfg=None, get_bigram_hash=lambda ids: torch.zeros_like(ids),
        examples=exs, tau=0.8, device="cpu", random_trials=10)
    res = out["result"]
    # mlp0 contributes nothing -> minimal circuit is {attn0}
    assert res.kept == ["attn0"], res.kept
    assert res.size == 1
    assert res.faithfulness >= 0.8
    # attn0 should carry far more attribution magnitude than mlp0
    assert abs(out["scores"]["attn0"]) > abs(out["scores"]["mlp0"]) + 1e-6


def test_measure_anchors_and_caching():
    model = TinyModel()
    exs = _examples()
    tp = P.TaskPatcher(model, None, lambda ids: torch.zeros_like(ids), exs, device="cpu",
                       nodes=["attn0", "mlp0"])
    model.requires_grad_(False)
    model._cd_enabled = True
    model._cd = tp.rec
    try:
        tp.compute_means()
        m_full = tp.measure(frozenset(["attn0", "mlp0"]))
        m_empty = tp.measure(frozenset())
        # mlp0 is a no-op node: ablating it must not change the metric
        m_no_mlp = tp.measure(frozenset(["attn0"]))
        assert abs(m_no_mlp - m_full) < 1e-5
        assert m_full != m_empty
        # cache returns identical object on second call
        assert tp.measure(frozenset()) == m_empty
    finally:
        model._cd_enabled = False
        model._cd = None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[ok] {name}")
    print("\nALL PATCHER TESTS PASSED")
