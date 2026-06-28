"""Capability-probe harness for the interpretability benchmark (Track 5).

Packs `ProbeExample`s into long token sequences (one forced-choice candidate per
EOT-delimited document) and scores them with the model's per-token loss, exactly
like `evals/hellaswag.py`. For each task we report forced-choice accuracy and a
logit-difference proxy (margin = loss(best wrong) - loss(correct)), and whether
the task clears the capability gate.

Only depends on the model's call interface
    model(input_seq, target_seq, seqlens, bigram_input_seq, schedule_cfg) -> loss_per_token
plus a `get_bigram_hash` callable and a `schedule_cfg`, so it never imports the
training script at module load.

The packing logic (`pack`, `_finalize`) is torch-only at score time; sequence
construction is pure Python and is unit-tested offline in `tests/`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from .probe_tasks import ProbeExample, build_all, tier_of

EOT_ID = 50256
PAD_ID = 0

# Forced-choice accuracy required for a task to "pass" the capability gate.
# Tasks below this are excluded from the benchmark (their circuits would be noise).
GATE_ACCURACY = {0: 0.0, 1: 0.80, 2: 0.65}  # by tier; control (tier 0) has no gate


@dataclass
class PackedExample:
    # span (start, end) into the packed targets for each candidate, plus label
    spans: list[tuple[int, int]]
    label: int
    task: str


@dataclass
class PackedSequence:
    inputs: Tensor          # int32 [seq_len]
    targets: Tensor         # int64 [seq_len]
    seqlens: Tensor         # int32 [max_num_docs] cumulative doc-end positions
    examples: list[PackedExample]


def _finalize(inputs: list[int], targets: list[int], examples: list[PackedExample],
              seq_len: int) -> PackedSequence:
    assert len(inputs) <= seq_len, f"sequence overflow: {len(inputs)} > {seq_len}"
    # document boundaries are positions right after each EOT separator
    doc_ends = [i + 1 for i, t in enumerate(inputs) if t == EOT_ID]
    pad = seq_len - len(inputs)
    inputs = inputs + [PAD_ID] * pad
    targets = targets + [PAD_ID] * pad

    # `seqlens` is handed directly to flash_attn_varlen as cu_seqlens, so it needs
    # the leading 0, every doc-end boundary, then padding entries (== seq_len, i.e.
    # trailing zero-length docs). Size from the ACTUAL doc count (our forced-choice
    # docs are ~tens of tokens, far shorter than hellaswag's, so the fixed
    # hellaswag formula underflows). Pad up to a multiple of 128 for shape sanity.
    n_boundaries = len(doc_ends) + 1  # +1 for the leading 0
    max_num_docs = (n_boundaries // 128 + 1) * 128
    cum = torch.full((max_num_docs,), seq_len, dtype=torch.int32)
    cum[0] = 0
    if doc_ends:
        de = torch.tensor(doc_ends, dtype=torch.int32)
        cum[1:len(de) + 1] = de
    return PackedSequence(
        inputs=torch.tensor(inputs, dtype=torch.int32),
        targets=torch.tensor(targets, dtype=torch.int64),
        seqlens=cum,
        examples=examples,
    )


def pack(examples: list[ProbeExample], seq_len: int) -> list[PackedSequence]:
    """Pack examples into sequences of `seq_len`.

    Layout per candidate document: <context><candidate>EOT. inputs drop the last
    token of each doc, targets drop the first, so the candidate span lines up
    with the positions that predict the candidate tokens.
    """
    sequences: list[PackedSequence] = []
    inputs: list[int] = []
    targets: list[int] = []
    packed: list[PackedExample] = []

    def flush():
        nonlocal inputs, targets, packed
        if inputs:
            sequences.append(_finalize(inputs, targets, packed, seq_len))
        inputs, targets, packed = [], [], []

    for ex in examples:
        total = sum(len(ex.context) + len(c) for c in ex.candidates)
        if len(inputs) + total + len(ex.candidates) >= seq_len:
            flush()
        spans: list[tuple[int, int]] = []
        for cand in ex.candidates:
            if inputs:  # EOT separator between docs (not at sequence start)
                inputs.append(EOT_ID)
                targets.append(EOT_ID)
            doc = ex.context + cand
            start = len(inputs) + len(ex.context) - 1  # -1: targets shifted left
            end = start + len(cand)
            spans.append((start, end))
            inputs.extend(doc[:-1])
            targets.extend(doc[1:])
        packed.append(PackedExample(spans=spans, label=ex.label, task=ex.task))
    flush()
    return sequences


@torch.no_grad()
def _score_sequence(model, schedule_cfg, seq: PackedSequence,
                    get_bigram_hash: Callable, device: str = "cuda") -> list[tuple[str, bool, float]]:
    bigram = get_bigram_hash(seq.inputs).to(device)
    loss_per_token = model(
        input_seq=seq.inputs.to(device),
        target_seq=seq.targets.to(device),
        seqlens=seq.seqlens.to(device),
        bigram_input_seq=bigram,
        schedule_cfg=schedule_cfg,
    )
    results = []
    for ex in seq.examples:
        cand_losses = [loss_per_token[s:e].mean().item() for s, e in ex.spans]
        pred = int(min(range(len(cand_losses)), key=lambda i: cand_losses[i]))
        correct = (pred == ex.label)
        # logit-diff proxy: best wrong candidate loss minus correct candidate loss
        correct_loss = cand_losses[ex.label]
        wrong_loss = min(l for i, l in enumerate(cand_losses) if i != ex.label)
        margin = wrong_loss - correct_loss
        results.append((ex.task, correct, margin))
    return results


@dataclass
class TaskReport:
    task: str
    tier: int
    n: int
    accuracy: float
    acc_stderr: float
    mean_margin: float
    passed: bool


def _aggregate(rows: list[tuple[str, bool, float]]) -> dict[str, TaskReport]:
    by_task: dict[str, list[tuple[bool, float]]] = {}
    for task, correct, margin in rows:
        by_task.setdefault(task, []).append((correct, margin))
    reports: dict[str, TaskReport] = {}
    for task, items in by_task.items():
        n = len(items)
        acc = sum(c for c, _ in items) / n
        stderr = math.sqrt(max(acc * (1 - acc), 0.0) / n)
        margin = sum(m for _, m in items) / n
        tier = tier_of(task)
        passed = acc >= GATE_ACCURACY.get(tier, 0.8)
        reports[task] = TaskReport(task, tier, n, acc, stderr, margin, passed)
    return reports


def run(model, schedule_cfg, seq_len: int, get_bigram_hash: Callable,
        print0: Callable, n_per_task: int = 200, seed: int = 0,
        tasks: list[str] | None = None, device: str = "cuda") -> dict[str, TaskReport]:
    """Generate, score, and report all probe tasks. Returns per-task reports."""
    t0 = time.perf_counter()
    model.eval()
    all_examples = build_all(n_per_task=n_per_task, seed=seed, tasks=tasks)

    rows: list[tuple[str, bool, float]] = []
    for task, examples in all_examples.items():
        seqs = pack(examples, seq_len=seq_len)
        for seq in seqs:
            rows.extend(_score_sequence(model, schedule_cfg, seq, get_bigram_hash, device=device))

    reports = _aggregate(rows)
    dt = time.perf_counter() - t0

    print0("=" * 72, console=True)
    print0(f"Track 5 capability probe  (n_per_task={n_per_task}, {dt:.1f}s)", console=True)
    print0(f"{'task':<18}{'tier':>5}{'n':>7}{'acc':>9}{'±se':>8}{'margin':>10}  gate", console=True)
    for task in sorted(reports, key=lambda t: (reports[t].tier, t)):
        r = reports[task]
        gate = "base" if r.tier == 0 else ("PASS" if r.passed else "fail")
        print0(f"{r.task:<18}{r.tier:>5}{r.n:>7}{r.accuracy:>9.3f}{r.acc_stderr:>8.3f}"
               f"{r.mean_margin:>10.3f}  {gate}", console=True)
    print0("=" * 72, console=True)
    return reports
