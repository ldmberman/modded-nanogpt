"""Offline tests for the model-independent circuit core (no torch needed).

Validates the greedy minimal-circuit search + backward prune + faithfulness
normalization against a synthetic *additive* model whose minimal circuit is known
analytically: each node adds a fixed contribution to the metric, and mean-ablation
zeroes that contribution, so M(kept) = base + sum(contrib[n] for n in kept).
"""

from __future__ import annotations

from records.track_5_interpretability import circuit as C
from records.track_5_interpretability import probe_tasks as T


def _make_measure(contrib: dict[str, float], base: float = 1.0):
    calls = {"n": 0}

    def measure(kept):
        calls["n"] += 1
        return base + sum(contrib[n] for n in kept)

    return measure, calls


def test_faithfulness_anchors():
    contrib = {"a": 3.0, "b": 1.0, "c": 0.0}
    measure, _ = _make_measure(contrib)
    nodes = list(contrib)
    m_full = measure(frozenset(nodes))
    m_empty = measure(frozenset())
    assert C.faithfulness(m_full, m_empty, m_full) == 1.0
    assert C.faithfulness(m_empty, m_empty, m_full) == 0.0


def test_greedy_recovers_known_minimal_set():
    # a+b+c+d = 10; top-2 (a=6,b=3) give 0.9 of the signal.
    contrib = {"a": 6.0, "b": 3.0, "c": 0.7, "d": 0.3}
    measure, _ = _make_measure(contrib)
    res = C.greedy_minimal_circuit(list(contrib), scores=contrib, measure=measure, tau=0.8)
    # 0.6 (a) < 0.8, 0.9 (a,b) >= 0.8  -> minimal size 2
    assert res.size == 2
    assert set(res.kept) == {"a", "b"}
    assert res.faithfulness >= 0.8


def test_backward_prune_drops_redundant_with_bad_ranking():
    # Perfect circuit is {a}, which alone gives faithfulness 1.0, but the ranking
    # is adversarial (puts useless 'c','d' first). Greedy may overshoot; prune
    # must shrink back toward minimal while staying >= tau.
    contrib = {"a": 10.0, "b": 0.0, "c": 0.0, "d": 0.0}
    bad_scores = {"a": 0.1, "b": 1.0, "c": 0.9, "d": 0.8}  # 'a' ranked last
    measure, _ = _make_measure(contrib)
    res = C.greedy_minimal_circuit(list(contrib), scores=bad_scores, measure=measure, tau=0.9, prune=True)
    assert "a" in res.kept
    assert res.faithfulness >= 0.9
    # prune removes the zero-contribution nodes
    assert res.size == 1


def test_negative_node_is_pruned():
    # 'neg' hurts the metric; greedy by |score| will try it, prune should drop it.
    contrib = {"a": 5.0, "b": 4.0, "neg": -2.0, "z": 0.0}
    scores = {"a": 5.0, "b": 4.0, "neg": 3.0, "z": 0.0}  # |neg| ranked high
    measure, _ = _make_measure(contrib)
    res = C.greedy_minimal_circuit(list(contrib), scores=scores, measure=measure, tau=0.8, prune=True)
    assert "neg" not in res.kept
    assert res.faithfulness >= 0.8


def test_random_baseline_below_targeted():
    contrib = {"a": 6.0, "b": 3.0, "c": 0.7, "d": 0.3}
    measure, _ = _make_measure(contrib)
    mean, std = C.random_faithfulness_at_size(list(contrib), measure, size=2, trials=50, seed=1)
    # average random-2 faithfulness should be well below the targeted top-2 (0.9)
    assert mean < 0.9


def test_build_cd_inputs_from_probe():
    import random
    exs = T.ioi(5, random.Random(0))
    cds = C.build_cd_inputs(exs)
    assert len(cds) == 5
    for cd, ex in zip(cds, exs):
        assert cd.read_pos == len(ex.context) - 1
        assert cd.correct_id == ex.candidates[ex.label][0]
        assert cd.distractor_id == ex.candidates[1 - ex.label][0]
        assert cd.corrupt_context is not None  # ioi carries a minimal pair


def test_geomean():
    assert abs(C.geomean([2, 8]) - 4.0) < 1e-9


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[ok] {name}")
    print("\nALL CIRCUIT CORE TESTS PASSED")
