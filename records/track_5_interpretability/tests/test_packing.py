"""Offline checks for Track 5 task generation + packing (CPU, no GPU model).

Run from repo root:
    .venv311/bin/python -m records.track_5_interpretability.tests.test_packing
"""

from __future__ import annotations

import torch

from records.track_5_interpretability import probe_tasks as T
from records.track_5_interpretability import probe_harness as H


def test_examples_wellformed():
    data = T.build_all(n_per_task=50, seed=1)
    for task, exs in data.items():
        assert exs, f"{task} produced no examples"
        for ex in exs:
            assert 0 <= ex.label < len(ex.candidates)
            assert len(ex.candidates) >= 2
            assert all(len(c) >= 1 for c in ex.candidates)
            assert len(ex.context) >= 1
            # correct and best-wrong candidate must differ in tokens
            assert ex.candidates[ex.label] != ex.candidates[1 - ex.label]
    print("[ok] examples well-formed for tasks:", ", ".join(data))


def test_spans_align_with_targets():
    """The crux: packed targets[span] must equal the candidate token ids."""
    data = T.build_all(n_per_task=40, seed=2)
    flat = [ex for exs in data.values() for ex in exs]
    seqs = H.pack(flat, seq_len=4096)
    assert seqs, "packing produced no sequences"

    # rebuild a flat index of (example, packed) pairs in order
    idx = 0
    for seq in seqs:
        # the i-th PackedExample corresponds to flat[idx+i]
        for j, pex in enumerate(seq.examples):
            ex = flat[idx + j]
            for cand_i, (s, e) in enumerate(pex.spans):
                got = seq.targets[s:e].tolist()
                want = ex.candidates[cand_i]
                assert got == want, (
                    f"{ex.task}: span {cand_i} targets {got} != candidate {want}"
                )
        idx += len(seq.examples)
    assert idx == len(flat)
    print(f"[ok] spans align with targets across {len(flat)} examples in {len(seqs)} seqs")


def test_doc_boundaries_and_padding():
    data = T.build_all(n_per_task=10, seed=3, tasks=["ioi"])
    seqs = H.pack(data["ioi"], seq_len=2048)
    for seq in seqs:
        assert seq.inputs.shape[0] == 2048
        assert seq.targets.shape[0] == 2048
        # seqlens is non-decreasing and starts at 0
        sl = seq.seqlens.tolist()
        assert sl[0] == 0
        assert all(sl[i] <= sl[i + 1] for i in range(len(sl) - 1))
        # number of EOT separators == (#docs - 1) within the used region
        n_eot = int((seq.inputs == H.EOT_ID).sum())
        n_docs = sum(len(e.spans) for e in seq.examples)
        assert n_eot == n_docs - 1, f"{n_eot} EOT vs {n_docs} docs"
    print("[ok] doc boundaries + padding consistent")


def test_mock_scoring():
    """A mock model that returns 0 loss exactly on the correct candidate tokens
    should yield 100% accuracy; the inverse should yield 0%."""
    data = T.build_all(n_per_task=30, seed=4, tasks=["induction", "successor_days"])
    flat = [ex for exs in data.values() for ex in exs]
    seqs = H.pack(flat, seq_len=4096)

    # Build a ground-truth "correct target id per position" map per sequence by
    # marking, for each example, the correct candidate's span positions.
    def uniform_model(input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg):
        return torch.ones_like(target_seq, dtype=torch.float32)

    # With uniform loss, argmin ties -> picks index 0, which is the correct label
    # for these tasks, so accuracy should be 1.0.
    rows = []
    for seq in seqs:
        rows.extend(H._score_sequence(uniform_model, None, seq,
                                      get_bigram_hash=lambda x: x, device="cpu"))
    reports = H._aggregate(rows)
    for task, r in reports.items():
        assert r.accuracy == 1.0, f"{task} acc {r.accuracy}"

    # Now a model that returns LOW loss only on the *wrong* candidate positions
    # should drive accuracy to 0 -> confirms scoring direction.
    def make_wrong_preferring(seq: H.PackedSequence):
        low = torch.ones(seq.targets.shape[0], dtype=torch.float32)
        for pex in seq.examples:
            wrong_i = 1 - pex.label
            s, e = pex.spans[wrong_i]
            low[s:e] = 0.0
        def model(input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg):
            return low
        return model

    rows = []
    for seq in seqs:
        rows.extend(H._score_sequence(make_wrong_preferring(seq), None, seq,
                                      get_bigram_hash=lambda x: x, device="cpu"))
    reports = H._aggregate(rows)
    for task, r in reports.items():
        assert r.accuracy == 0.0, f"{task} acc {r.accuracy} (expected 0)"
    print("[ok] mock scoring path runs, aggregates, and respects loss direction")


if __name__ == "__main__":
    test_examples_wellformed()
    test_spans_align_with_targets()
    test_doc_boundaries_and_padding()
    test_mock_scoring()
    print("\nALL TRACK 5 PACKING TESTS PASSED")
