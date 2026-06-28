"""Circuit discovery + minimal-circuit-size scoring for Track 5.

This module is the *model-independent* core: it defines the metric, the
faithfulness normalization, and the greedy minimal-circuit search with a backward
prune, plus the random-circuit baseline. It operates entirely through two
callbacks supplied by the model-coupled layer (`Patcher`, added separately):

    measure(kept: frozenset[str]) -> float
        Mean task metric M over the examples when every node NOT in `kept` is
        mean-ablated (frozen to its distribution-mean activation). `kept == all`
        gives M_full (nothing ablated); `kept == {}` gives M_empty (all nodes
        ablated).

    scores: dict[node_name -> float]
        Attribution-patching ranking signal (sign-agnostic magnitude is used for
        ordering). AP only *ranks*; faithfulness is always *measured* via
        `measure`, never read off the AP estimate.

Faithfulness:  F(C) = (measure(C) - M_empty) / (M_full - M_empty).
Minimal circuit size at target tau:  smallest |C| with F(C) >= tau, found by
greedy addition in |score| order then a backward prune.

Design notes are torch-free so this file is unit-tested on CPU; see tests/.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field
from typing import Callable

from .probe_tasks import ProbeExample


@dataclass
class CDExample:
    """One single-sequence circuit-discovery item (not packed).

    context: token ids fed as `input_seq`; logits at position len-1 predict the
        answer. correct/distractor are single token ids; the metric is their
        logit difference. corrupt_* carry the optional minimal pair (used for
        resample-ablation cross-checks; mean-ablation does not need it).
    """

    context: list[int]
    read_pos: int
    correct_id: int
    distractor_id: int
    task: str
    corrupt_context: list[int] | None = None
    meta: dict = field(default_factory=dict)


def build_cd_inputs(examples: list[ProbeExample]) -> list[CDExample]:
    """Convert forced-choice ProbeExamples (single-token candidates) to CDExamples."""
    out: list[CDExample] = []
    for ex in examples:
        if len(ex.candidates) != 2 or any(len(c) != 1 for c in ex.candidates):
            raise ValueError(f"circuit discovery expects 2 single-token candidates; got {ex.candidates}")
        correct = ex.candidates[ex.label][0]
        distractor = ex.candidates[1 - ex.label][0]
        out.append(CDExample(
            context=list(ex.context),
            read_pos=len(ex.context) - 1,
            correct_id=correct,
            distractor_id=distractor,
            task=ex.task,
            corrupt_context=list(ex.corrupt_context) if ex.corrupt_context is not None else None,
            meta=dict(ex.meta),
        ))
    return out


def faithfulness(m: float, m_empty: float, m_full: float) -> float:
    denom = m_full - m_empty
    if abs(denom) < 1e-9:
        return float("nan")
    return (m - m_empty) / denom


@dataclass
class CircuitResult:
    task: str
    kept: list[str]
    size: int
    faithfulness: float
    m_full: float
    m_empty: float
    tau: float
    curve: list[tuple[int, float]]  # (size, faithfulness) along greedy addition


def greedy_minimal_circuit(
    nodes: list[str],
    scores: dict[str, float],
    measure: Callable[[frozenset[str]], float],
    tau: float,
    task: str = "",
    prune: bool = True,
) -> CircuitResult:
    """Greedy add by |score| until measured faithfulness >= tau, then backward-prune.

    `measure` is assumed deterministic and is memoized by the caller if costly.
    """
    all_nodes = frozenset(nodes)
    m_full = measure(all_nodes)
    m_empty = measure(frozenset())

    ordered = sorted(nodes, key=lambda n: abs(scores.get(n, 0.0)), reverse=True)

    kept: list[str] = []
    curve: list[tuple[int, float]] = [(0, faithfulness(m_empty, m_empty, m_full))]
    reached = False
    for n in ordered:
        kept.append(n)
        f = faithfulness(measure(frozenset(kept)), m_empty, m_full)
        curve.append((len(kept), f))
        if f >= tau:
            reached = True
            break

    if prune and reached:
        # Try removing nodes (least-attributed first) while staying >= tau.
        for n in sorted(list(kept), key=lambda n: abs(scores.get(n, 0.0))):
            trial = [k for k in kept if k != n]
            f = faithfulness(measure(frozenset(trial)), m_empty, m_full)
            if f >= tau:
                kept = trial

    final_f = faithfulness(measure(frozenset(kept)), m_empty, m_full)
    return CircuitResult(task=task, kept=kept, size=len(kept), faithfulness=final_f,
                         m_full=m_full, m_empty=m_empty, tau=tau, curve=curve)


def random_faithfulness_at_size(
    nodes: list[str],
    measure: Callable[[frozenset[str]], float],
    size: int,
    trials: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """Mean +/- std faithfulness of a random `size`-node circuit (the noise floor)."""
    rng = _random.Random(seed)
    m_full = measure(frozenset(nodes))
    m_empty = measure(frozenset())
    vals: list[float] = []
    for _ in range(trials):
        pick = frozenset(rng.sample(nodes, size)) if size <= len(nodes) else frozenset(nodes)
        vals.append(faithfulness(measure(pick), m_empty, m_full))
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var)


def geomean(sizes: list[int]) -> float:
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return 0.0
    return math.exp(sum(math.log(s) for s in sizes) / len(sizes))
