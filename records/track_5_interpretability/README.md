# Modded-NanoGPT Interpretability Benchmark (Track 5)

> Status: v0 done, v1 in progress. Numbers marked _TBD_ are unset until we have enough empirical data to pin them.

## Motivation

[Gao et al. 2025](https://arxiv.org/pdf/2511.13653) show that weight-sparse transformers form task circuits roughly 16× smaller than dense models at matched loss, at the cost of a 100–1000× training-compute penalty. Since real-world incentives push models to be fast, interpretability research confined to slow, sanitized models risks irrelevance. This track asks the inverse question: how much of the dense↔sparse compute–interpretability gap is closable, starting from the fastest known training recipes?

We take the Track 1 speedrun recipe (GPT-2-small scale: `num_layers=11, num_heads=6, head_dim=128, model_dim=768`, FineWeb, ≤3.28 val loss) as the cheap, capable, illegible anchor and measure how far interpretability can be pushed — and at what compute cost — without abandoning the speed-oriented architecture.

## Benchmark definition

A submission is a model + training run + the circuits discovered on it. Validity rules:

1. Capability gate: ≤3.28 mean FineWeb val loss under the repo's standard significance protocol (p < 0.01). Capability is a gate, not a scored axis — this blocks "interpretable because it does nothing" solutions.
2. Fixed data pipeline: train/val token streams unchanged (same rule as the main speedrun). Batch size, sequence length, architecture, and sparsity machinery are free.
3. Reproducible circuits: release the checkpoint, seeds, and discovered circuits so the metric can be re-computed.

Two scored axes at fixed loss:

- Compute, in training tokens (hardware-independent, auditable).
- Interpretability: geometric-mean circuit size across the gated task suite (lower is better).

The leaderboard is the Pareto frontier of (compute, interpretability): a submission earns a record if it is non-dominated. Track 1 records automatically hold the cheap corner of the frontier. We deliberately do not collapse the two axes into a ratio: a ratio fixes an arbitrary exchange rate between incommensurate units, has no quality floor, and is statistically ill-behaved; it may be reported as a descriptive secondary stat only.

Statistical dominance (thresholds _TBD_ from measured metric variance): a new point must beat the frontier on one axis at p < 0.01 while being no worse than noise on the other, with a minimum-improvement margin to keep the frontier from filling with trivial points.

## Interpretability metric

We use the metric definition from Gao et al. — circuit size at a target task faithfulness, mean-ablation convention — but discover circuits cheaply via attribution patching instead of per-task mask training.

### Task suite

Tasks are forced-choice with single-token answers and are admitted only if the anchor performs them above threshold (tier 1 ≥ 0.80, tier 2 ≥ 0.65 accuracy); a circuit for a task the model cannot do is noise. Measured capabilities of the v0 anchor are in [Appendix A](#appendix-a--v0-capability-probe-results-track-1-anchor). The gated-PASS suite: induction, ioi_copy, successor_{days,months,digits,letters}, bracket_d0, factual_recall, greater_than, pronoun_gender, subject_verb_agreement. Failing tasks (IOI family, verbatim recall, deep brackets, addition) are kept as held-out diagnostics: they share a missing inhibition/binding mechanism, and their appearance in future submissions is a key signal.

### Circuit discovery and faithfulness control

The core rule: attribution patching only ranks candidate nodes; faithfulness is always measured by a real mean-ablation forward pass, never read off the linear AP estimate. Per task:

- Metric: `M = logit(correct) − logit(distractor)` at the answer position (raw `lm_head`).
- Faithfulness of a circuit `C`: `F(C) = (M(C) − M_∅) / (M_full − M_∅)`, where `M(C)` mean-ablates every node outside `C` and `M_∅` ablates all nodes.
- Circuit size: smallest `|C|` with measured `F ≥ τ` (working value `τ = 0.8`), found by greedy addition in `|AP score|` order, then a backward prune so ranking error cannot inflate size.

Integrity baselines:

- Random-circuit baseline: same-size random node sets. The AP-vs-random faithfulness gap is the signal-to-noise; overlap means the "circuit" is meaningless.
- Control task (`control_random`): the full pipeline on a structureless task sets the uninterpretable floor.
- Planned upgrades: EAP-IG on edges with saturation/zero-gradient artifacts; a mask-based structured-pruning spot check on 2–3 tasks as a gold standard.

### Node graph

v1 granularity: the component-level residual writers exposed by the fused forward — `attn{i}` (i≠6), `mlp{i}`, `skip6` — 22 ablatable nodes; embeddings, bigram, value-embed, and MUDD scaffolding stay always-on. Per-head edges and the architecture-specific injection paths (value embeddings, MUDD skips, bigram subspace, smear, XSA) as first-class nodes require an eager attention reimplementation (FA3 hides head internals); this is the next refinement and should shrink and spread sizes. Whether the speed-shortcut paths dominate circuits is itself an interpretability finding.

### Aggregation

- Primary: geometric-mean circuit size over the gated suite at fixed `τ`.
- Secondary (planned): a cross-task hub metric. Overlay per-task circuits; penalize high-degree nodes weighted by polysemanticity, measured by Ablation-Influence Similarity: `AIS(v)` = mean over task pairs of the cosine between `v`'s ablation-induced downstream-influence vectors. High AIS (consistent footprint) marks a reusable monosemantic hub, exempt from penalty; low AIS marks a polysemantic hub, penalized in proportion to degree (`hub_penalty = Σ degree(v)·(1 − AIS(v))` over top-k nodes). AIS is a proxy and must be validated on knowns first (induction head high, a polysemantic MLP hub low, control task as null); open choices include one-hop vs all-downstream influence and the penalty weighting.

## Running

The harness lives in this directory; hooks in `train_gpt.py` are env-gated and default-off (a class-constant gate lets `torch.compile` fold the taps out of the training graph).

Capability probe (`probe_tasks.py` + `probe_harness.py`; single GPU, no data files needed):

```bash
TRACK5_PROBE_CKPT=logs/<run_id>/state_step001390.pt \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Knobs: `TRACK5_PROBE_N` (examples/task, default 200), `TRACK5_PROBE_SEQLEN` (default 16384). Examples are seeded deterministically (`crc32`), so runs are reproducible. To produce a checkpoint, set `save_checkpoint: bool = True` in `Hyperparameters` of the pinned Track 1 `train_gpt.py`; the `.pt` stores the state dict and source code.

Circuit discovery (`circuit.py` = model-independent search/faithfulness core; `circuit_patcher.py` = tap protocol, mean/clean/patch passes):

```bash
TRACK5_PROBE_CKPT=logs/<run_id>/state_step001390.pt TRACK5_CIRCUIT=1 \
TRACK5_CIRCUIT_TASKS=induction,ioi_copy,successor_days,... \
TRACK5_CD_N=32 TRACK5_CD_TAU=0.8 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Offline tests (CPU): `tests/test_packing.py`, `tests/test_circuit.py`, `tests/test_patcher.py`.

## Status and roadmap

Done (v0): trained a Track 1 checkpoint to 3.2798; capability probe over 20 tasks ([Appendix A](#appendix-a--v0-capability-probe-results-track-1-anchor)); node attribution-patching circuit discovery with measured-faithfulness selection and random baselines ([Appendix B](#appendix-b--v0-circuit-discovery-results-track-1-anchor)). Geomean circuit size 13.97 at `τ = 0.8` over 11 gated-PASS tasks; AP circuits far above the random floor on every task.

v1 (in progress):

- Per-head / injection-node granularity via an eager attention reimplementation.
- Control-task floor and held-out suite wired into the aggregate; faithfulness-vs-size curves; EAP-IG where saturation artifacts appear.
- Interpretability across training: run the metric on checkpoints along a single training run to test whether circuit complexity improves in phase transitions — the hypothesis being that models start messy and compress into cleaner mechanisms as they generalize. Requires periodic checkpointing; the metric run is cheap (~5 GPU-min/checkpoint).
- Sweep the pinned Track 1 record checkpoints and plot how interpretability changed as the speedrun got faster.
- Joint interactive visualization: all task circuits overlaid on one plot with per-task highlighting.

v1.5: the AIS hub classifier and connectedness metric, validated on known reusable vs polysemantic hubs before it weights anything.

v2: open the frontier leaderboard; first interpretability-oriented submissions vs the Track 1 anchor. Candidate submission strategy — circuit seeding: hand-initialize known mechanisms the anchor lacks (e.g. S-inhibition head pairs for IOI) in the untrained model and train with them, testing whether seeded circuits survive training, speed it up, and land as small verified circuits.

Open decisions: dominance thresholds and minimum-improvement margin; final `τ` and aggregation constants; final admitted suite; whether EAP-IG is needed from the start; AIS specifics.

## Appendix A — v0 capability-probe results (Track 1 anchor)

Checkpoint `logs/9c4d9aab-b531-4158-9575-4ddb001770e7/state_step001390.pt`, val loss 3.2798. Probe: n=200/task, forced choice by mean per-token loss; margin = loss(best wrong) − loss(correct). Gates: tier 1 ≥ 0.80, tier 2 ≥ 0.65. The PASS/fail gate is the robust quantity to pin across runtimes; raw margins vary mildly.

| task | tier | acc | ±se | margin | gate |
|---|---|---|---|---|---|
| ioi_copy | 1 | 1.000 | 0.000 | 8.73 | PASS |
| successor_days | 1 | 1.000 | 0.000 | 7.74 | PASS |
| successor_digits | 1 | 1.000 | 0.000 | 9.05 | PASS |
| successor_months | 1 | 1.000 | 0.000 | 4.78 | PASS |
| induction | 1 | 0.895 | 0.022 | 6.92 | PASS |
| successor_letters | 1 | 0.865 | 0.024 | 2.98 | PASS |
| bracket_d0 | 1 | 0.830 | 0.027 | 6.93 | PASS |
| bracket_d1 | 1 | 0.585 | 0.035 | 2.98 | fail |
| factual_recall | 2 | 1.000 | 0.000 | 9.47 | PASS |
| subject_verb_agreement | 2 | 1.000 | 0.000 | 4.24 | PASS |
| pronoun_gender | 2 | 0.980 | 0.010 | 4.26 | PASS |
| greater_than | 2 | 0.800 | 0.028 | 1.37 | PASS |
| bracket_d2 | 2 | 0.560 | 0.035 | 1.87 | fail |
| bracket_d3 | 2 | 0.540 | 0.035 | 0.97 | fail |
| single_digit_addition | 2 | 0.465 | 0.035 | −0.41 | fail |
| bracket_d4 | 2 | 0.470 | 0.035 | 0.18 | fail |
| ioi | 2 | 0.445 | 0.035 | −0.24 | fail |
| verbatim_recall | 2 | 0.350 | 0.034 | −0.70 | fail |
| ioi_oneshot | 2 | 0.110 | 0.022 | −2.21 | fail |
| control_random | 0 | 0.515 | 0.035 | 0.13 | (chance) |

The split is mechanistically coherent. The model is strong wherever a task reduces to copy, induction, sequence continuation, local syntax, or lookup; it fails wherever it must select a structurally-correct token over a more salient (recent or duplicated) competitor. Three signatures converge on a missing inhibition/binding mechanism:

1. IOI ladder: ioi_copy 1.00 → ioi 0.45 → ioi_oneshot 0.11. Copying is intact; the moment a duplicated subject competes, the model follows it, and a worked demonstration makes it worse (induction latches onto the salient repeat). The missing piece is S-inhibition specifically.
2. Verbatim recall is below chance (0.35): the model copies the most recent value instead of binding the queried key, dissociating from induction (0.90).
3. Bracket margins decay monotonically with depth (6.93 → 0.18, d0→d4): it closes a lone pair but cannot bind a closer to the innermost opener once an enclosing bracket competes.

Implication: a fast-trained model readily acquires copying, induction, retrieval, and knowledge, but not the inhibition/binding circuits the interpretability literature treats as the canonical clean multi-step circuits (IOI being the flagship). The concrete question for the track: does extra compute or sparsity buy the binding/inhibition circuits the dense anchor lacks? The failing tasks are where that signal appears first.

## Appendix B — v0 circuit-discovery results (Track 1 anchor)

Same checkpoint as Appendix A. Node attribution patching over the 22 component-level writers; greedy add by |score| to measured faithfulness τ = 0.8, then backward prune; n = 32 examples/task. `M_full` = mean clean logit difference; `F` = measured faithfulness; `rand@size` = faithfulness of a same-size random circuit (mean ± std).

| task | circuit size | F (measured) | M_full | rand @ size |
|---|---|---|---|---|
| ioi_copy | 11 | 0.937 | +23.28 | 0.09 ± 0.13 |
| successor_letters | 12 | 0.884 | +4.68 | 0.08 ± 0.10 |
| bracket_d0 | 13 | 0.822 | +9.14 | 0.10 ± 0.07 |
| induction | 14 | 1.238 | +12.93 | −0.01 ± 0.12 |
| successor_days | 14 | 0.826 | +11.58 | 0.12 ± 0.13 |
| successor_months | 14 | 0.811 | +8.54 | 0.15 ± 0.12 |
| factual_recall | 14 | 0.864 | +15.62 | 0.10 ± 0.10 |
| pronoun_gender | 14 | 0.802 | +11.18 | 0.08 ± 0.14 |
| subject_verb_agreement | 14 | 0.906 | +6.33 | −0.01 ± 0.05 |
| successor_digits | 17 | 0.813 | +15.36 | 0.21 ± 0.21 |
| greater_than | 18 | 0.891 | +3.08 | 0.14 ± 0.29 |

Geometric-mean circuit size: 13.97 over the 11 gated-PASS tasks.

Reading the numbers:

- AP ≫ random everywhere (random floors −0.01 to 0.21 vs AP ≥ 0.8), so the size signal is real.
- Sizes are large in absolute terms (11–18 of 22) because the graph is component-level; per-head granularity should shrink and spread them. The absolute geomean is an anchor baseline — the leaderboard cares about relative movement.
- Induction's F = 1.24 means the circuit over-recovers: some excluded components are suppressors whose ablation raises the metric. Reported as-is, not clipped.
- Capability strength ≠ circuit size: ioi_copy is the strongest behavior (+23) with the smallest circuit (11); greater_than is the weakest (+3.1) with the largest, most diffuse (18). Diffuse-and-weak is the profile interpretable training should tighten.

## References

- [Gao et al., "Weight-sparse transformers have interpretable circuits," arXiv:2511.13653 (2025).](https://arxiv.org/pdf/2511.13653) ([code](https://github.com/openai/circuit_sparsity/))
- [Hanna et al., "Have Faith in Faithfulness: ... EAP-IG," arXiv:2403.17806 (2024).](https://arxiv.org/abs/2403.17806)
- [Wang et al., "Interpretability in the Wild: ... IOI circuit in GPT-2 small," arXiv:2211.00593 (2022).](https://arxiv.org/abs/2211.00593)
- [Olsson et al., "In-context Learning and Induction Heads," (2022).](https://arxiv.org/abs/2209.11895)
