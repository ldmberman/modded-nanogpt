# Modded-NanoGPT Interpretability Benchmark (Track 5)

> **Status: v0 spec / draft.** This document defines the benchmark we are converging on. Numbers marked _TBD_ are intentionally unset until the first empirical pass on the Track 1 anchor tells us the realistic ranges and metric variance. Expect this spec to change.

## Motivation

[Gao et al. 2025, "Weight-sparse transformers have interpretable circuits"](https://arxiv.org/pdf/2511.13653) show that constraining most of a transformer's weights to zero (`L0` weight sparsity) yields models whose task circuits are dramatically more human-understandable — roughly 16× smaller circuits than a dense model at matched pretraining loss. The catch is a large training-compute penalty: weight-sparse models are extremely inefficient to train, and the paper notes a 100–1000×-scale gap, with interpretability hard to preserve beyond tens of millions of nonzero parameters.

Since there is a strong real-world incentive to make models *fast*, studying interpretability only on clean, slow, sanitized models risks being irrelevant to the models people actually deploy. This track asks the inverse question:

> **How much of the dense↔sparse compute–interpretability gap is closable, starting from the fastest known training recipes?**

Concretely: take the speedrun's highly-optimized recipe as the cheap, capable, illegible anchor, and see how far we can push interpretability — and at what compute cost — *without* abandoning the speed-oriented architecture. Shrinking this gap is plausibly useful for AI safety: cheaper interpretable models are more likely to actually be used.

## The core open question

Can a fast-trained model (Track 1 lineage, GPT-2-small scale: `num_layers=11, num_heads=6, head_dim=128, model_dim=768`, FineWeb, ≤3.28 val loss) form *sufficiently clean circuits* across a diverse task suite to support a meaningful yet affordable interpretability metric? We begin by checking whether it forms a clean **induction** circuit, then expand the task suite.

## Benchmark definition (tl;dr)

A submission is a **model + training run + the circuits discovered on it**. It is valid if:

1. **Capability gate.** Attains ≤ 3.28 mean FineWeb val loss, with the repo's standard statistical-significance protocol (enough run logs for p < 0.01 that mean val loss ≤ 3.28).
2. **Fixed data pipeline.** Does not modify the train/val token streams (same rule as the main speedrun). Batch size, sequence length, architecture, and sparsity machinery are all free.
3. **Reproducible circuits.** Releases the checkpoint, the exact token range/seed used, and the discovered circuits, so the interpretability metric can be re-computed.

Capability is a **gate, not a scored axis** — this is what prevents degenerate "interpretable because it does nothing" solutions.

## The two scored axes

At fixed loss we have a genuine 2-D trade-off, so we do **not** collapse it to a single number (no `loss×time`, no `metric/compute` ratio — see [Why no ratio](#why-not-a-single-ratio-metric)).

1. **Compute** — measured in **training tokens** (hardware-independent, auditable, matches the repo's framing).
2. **Interpretability** — geometric-mean circuit size across the task suite (lower is better); see [Interpretability metric](#interpretability-metric).

## Leaderboard = the Pareto frontier

There is **no single champion.** The leaderboard is the Pareto frontier of (compute, interpretability) at fixed loss. A submission earns a record if it is **non-dominated**: either more interpretable than anything at its compute, or cheaper (fewer tokens) than anything at its interpretability. You earn a place by *pushing the frontier out*.

This is the same object the paper reports (its capability–interpretability frontier), and it resolves two design traps:

- **No cramped single budget.** Within any compute region, interpretability has full headroom to improve.
- **No unbounded compute.** A point only counts if it advances the frontier; throwing 10,000× compute for a marginal gain does not dominate cheaper points.

### Relationship to Track 1 (the anchor)

The Track 1 record (fast, capable, illegible) is a valid submission at the **bottom-left** of the frontier: minimal tokens, poor interpretability. It can only ever own the *cheap corner*. A "more expensive but more interpretable" submission is non-dominated and sits higher on the frontier — Track 1 cannot dominate it because it is strictly worse on the interpretability axis. So "does the latest Track 1 record win here?" has a clean answer: it is the champion of the cheapest region only, and a new Track 1 record simply *raises the floor* that interpretability work must beat. The coupling is a feature — Track 1 hands us a calibrated dense/illegible anchor for free.

### Statistical dominance (open: thresholds _TBD_)

Because the interpretability metric is stochastic, "non-dominated" needs a statistical definition. Working rule:

- A new point must beat the current frontier on **one** axis at **p < 0.01** while being **no worse than noise** on the other.
- A **minimum-improvement threshold** keeps the frontier from filling with trivial points.

The exact thresholds depend on the metric's measured variance, which the v0 anchor experiment will reveal.

## Interpretability metric

We borrow the paper's metric *definition* (circuit size at a target task faithfulness, mean-ablation convention) so our numbers stay comparable, but discover circuits cheaply via attribution rather than per-task mask-training.

### Task suite

Empirically gated: a task is only admitted if the checkpoint actually performs it above threshold (else its "circuit" is noise). Candidate tiers for a GPT-2-small FineWeb model:

- **Tier 1 (high confidence):** induction (random repeated-token copying), previous-token / duplicate-token, successor (numbers/days/months).
- **Tier 2 (likely, well-documented on GPT-2-small):** IOI, greater-than (years), gendered-pronoun / simple coreference, subject–verb number agreement.
- **Tier 3 (stretch, expect messy/failed):** single-digit addition, acronyms. Treated as research probes, not benchmark pillars.

A **held-out task suite** is reported for benchmark health (detecting overfitting/gaming of the metric).

Measured results on the v0 anchor are in [Appendix A](#appendix-a--v0-capability-probe-results-track-1-anchor): the anchor passes induction, copy, successors, agreement, gender-pronoun, factual recall, greater-than, and shallow brackets, but **fails the entire IOI family, verbatim/key-binding, deep brackets, and arithmetic** — a coherent "copy/lookup yes, inhibition/binding no" profile.

### Circuit discovery

- **Start: plain attribution patching** (first-order clean−corrupt estimate). Cheap; validate the harness on induction (cross-checked against a direct attention-pattern induction score) and reproduce a known IOI circuit before trusting aggregates.
- **Upgrade: [EAP-IG](https://arxiv.org/abs/2403.17806)** (integrated-gradients edge attribution) only on edges showing saturation / zero-gradient artifacts (the non-linear hubs).
- **Gold-standard spot check:** the paper's mask-based structured pruning on 2–3 tasks, to validate faithfulness — not for the aggregate.

### Graph definition over the speedrun architecture

Attribution patching is general over any differentiable graph, but the speedrun model has non-standard paths that **must be explicit, labeled nodes** (never silently merged), since the hub metric depends on it:

- Standard: per-layer attention head outputs, MLP outputs, residual reads/writes.
- Architecture-specific: value-embedding injection points, MUDD skips (embed→block, block 3→6), the bigram-hash subspace (1/4 of `model_dim`), the smear (1-token lookback), XSA-gated paths.

Whether these speed-shortcut paths dominate the circuits is itself a headline interpretability finding.

### Aggregation

- **Primary:** geometric-mean number of edges across the task suite, at a fixed per-task faithfulness target (logit-difference recovery ≥ _TBD_, mean-ablation).
- **Secondary (novel):** a cross-task **hub / connectedness** metric — overlay per-task circuits and reward absence of hubs, *except* hubs that are consistently reused. Good vs bad hubs are separated by the **Ablation-Influence Similarity (AIS)** metric defined below.

### Hub classification via Ablation-Influence Similarity (AIS)

The connectedness metric must *not* blindly penalize high-degree nodes: some hubs are legitimately reused, monosemantic components (an induction head used the same way across many tasks), while others are polysemantic bottlenecks that genuinely hurt interpretability. AIS is how we tell them apart.

**Definition.** Take the overlaid circuit graph and consider only the top-*k* highest-degree nodes (this step is expensive, so it is restricted). For a candidate hub node `v` and a task `t`:

1. **Mean-ablate** `v` (freeze its activation at the mean over the pretraining distribution, same convention as the circuit pruning), and run the task examples.
2. Record the **ablation-influence vector** `I[v,t]` — indexed by `v`'s downstream nodes `d`, where `I[v,t][d]` is the (signed, example-averaged) change in `d`'s activation caused by ablating `v`. The index set `d` is fixed by the model, so vectors are comparable across tasks.

Then:

```
AIS(v) = mean over task pairs (t, t') of  cosine( I[v,t], I[v,t'] )
```

**Interpretation.** Cosine is scale-invariant, so AIS measures *whether `v` reshapes downstream computation in the same pattern regardless of task*, not how strongly:
- **High AIS** ⇒ consistent downstream footprint ⇒ monosemantic, reusable ⇒ **exempt** from (or rewarded by) the hub penalty.
- **Low AIS** ⇒ task-dependent footprint ⇒ polysemantic hub ⇒ **penalized**.

**Integration with the connectedness metric.** The hub penalty is degree-weighted by polysemanticity, e.g.

```
hub_penalty = sum over top-k nodes v of  degree(v) * (1 - AIS(v))
```

so a reusable induction-head hub (AIS≈1) contributes ~0, while a polysemantic hub (low AIS) is penalized in proportion to its degree. The secondary connectedness score combines overlaid-graph connectedness with this penalty.

**Known caveats (must validate, not assume).** AIS is a *proxy*. A genuinely monosemantic node can still show a low cross-task cosine if it feeds *different downstream circuits* per task (its own function is fixed but its consumers differ); conversely two unrelated uses could coincidentally share a downstream footprint. Mitigations to evaluate:
- Restrict `d` to `v`'s **immediate (one-hop) readers** rather than all downstream nodes, to isolate `v`'s intrinsic function from downstream-circuit differences.
- Cross-check against an **activation/feature-direction** variant (is `v`'s output direction or attention pattern itself consistent across tasks?).

**Validation.** Before trusting AIS in the score, sanity-check it on knowns: the **induction head** (reused → expect high AIS) vs. a known polysemantic MLP hub (expect low AIS), with the control task as a null. Only adopt the penalty weighting once AIS separates these as predicted.

**Cost.** One pass to estimate mean activations, then ~`O(k × tasks)` ablate-and-measure forward passes. Reserved for the secondary metric, computed far less often than the primary circuit-size number.

## Metric integrity / anti-gaming

How faithfulness is controlled (the key guard): **attribution patching only *ranks* candidates; faithfulness is always *measured* by a real mean-ablation forward, never read off the AP estimate.** A circuit `C` is scored by freezing every node outside `C` to its distribution-mean activation and measuring `F(C) = (M(C) − M_∅) / (M_full − M_∅)`, where `M_∅` ablates all component nodes. Circuit *size* = smallest `|C|` with measured `F ≥ τ`, found by greedy addition in `|score|` order then a backward prune (so AP ranking error cannot inflate size). This makes AP's linear-approximation error irrelevant to the reported size: a node only counts if ablating it in actually moves the metric.

- **Faithfulness target:** working value `τ = 0.8` (logit-difference recovery), mean-ablation convention. On the anchor every gated-PASS task reaches it; see [Appendix B](#appendix-b--v0-circuit-discovery-results-track-1-anchor).
- **Random-circuit baseline:** for each task, compare against same-size random node sets. The attribution-vs-random gap is the signal-to-noise; overlap means the "circuit" is meaningless. On the anchor random circuits sit at `F ≈ 0.0–0.2` at the AP-circuit sizes while the AP circuits reach `≥ 0.8` — a wide, consistent gap.
- **Control task:** run the full pipeline on a shuffled-label / structureless task to set the empirical "uninterpretable" floor.
- The random/control baselines are the primary guard: if a model scores well on them, the metric is exploitable and must be revised.

## Why not a single ratio metric?

`interpretability / compute` looks clean but is strictly weaker than the frontier:

- It freezes an arbitrary, unit-dependent exchange rate (a line through the origin); change tokens↔FLOPs or add fixed overhead and the "winner" moves.
- The units (`edges / token`) aren't commensurate and assume a linear, scale-free trade at every operating point; the real frontier is curved.
- No quality floor — it crowns "mildly interpretable but very cheap" over "genuinely interpretable but costlier," which is wrong for a safety benchmark.
- Ratios of noisy quantities are heavy-tailed/biased, wrecking significance testing.

A ratio is fine only as a *descriptive secondary stat*, never the optimization target.

## Roadmap

- **v0 (bootstrap the metric). _Done._** Trained a Track 1 checkpoint to 3.2798 with `save_checkpoint=True`; built the capability probe ([Appendix A](#appendix-a--v0-capability-probe-results-track-1-anchor)) and the node attribution-patching circuit-size harness with measured-faithfulness selection + random baseline ([Appendix B](#appendix-b--v0-circuit-discovery-results-track-1-anchor)). Geomean size 13.97 at `τ=0.8` over 11 gated-PASS tasks; AP ≫ random throughout.
- **v1 (in progress).** Random-circuit baseline ✅ and faithfulness target (`τ=0.8`) ✅ are in. Remaining: per-head / injection-node granularity (eager attention reimpl), the control-task floor and held-out suite in the aggregate, faithfulness-vs-size curves, and EAP-IG on saturated edges.
- **v1.5 (AIS).** Build the overlaid-graph connectedness metric and the Ablation-Influence Similarity hub classifier; validate AIS on the induction head (expect high) vs a polysemantic MLP hub (expect low) before letting it weight the hub penalty.
- **v2.** Open the frontier leaderboard; first sparse / interpretability-oriented submissions vs the Track 1 anchor.

## Running the v0 capability probe

The probe is implemented as:

- `probe_tasks.py` — forced-choice task generators: induction, successor {days, months, digits, letters}, the IOI ladder {ioi_copy, ioi, ioi_oneshot}, greater-than, pronoun-gender, subject-verb agreement, verbatim-recall (key-value binding), single-digit addition, factual recall (capitals), bracket-matching {d0..d4}, plus a `control_random` chance baseline. Examples are seeded deterministically (`crc32`) so runs are reproducible. Torch-free, unit-tested on CPU.
- `probe_harness.py` — packs examples into EOT-delimited sequences and scores each candidate by mean per-token loss (mirrors `evals/hellaswag.py`), reporting forced-choice accuracy, a logit-difference proxy (margin), and the capability gate.
- An eval-only hook in `train_gpt.py` (guarded by `TRACK5_PROBE_CKPT`) that loads a checkpoint, replays the YaRN window schedule, runs the probe, and exits without compiling or training.

**Pinning + training a checkpoint.** Pick a Track 1 record (this pins the architecture/version for an interpretability record), use its `train_gpt.py`, set `save_checkpoint: bool = True` in `Hyperparameters`, and run the normal training command. The checkpoint is written to `logs/<run_id>/state_step<NNNNNN>.pt` (it stores both the model `state_dict` and the source `code`).

**Probing the checkpoint** (single GPU is fine; no data files needed — the hook exits before the data loaders are built):

```bash
TRACK5_PROBE_CKPT=logs/<run_id>/state_step001390.pt \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Optional env knobs: `TRACK5_PROBE_N` (examples per task, default 200), `TRACK5_PROBE_SEQLEN` (packing length, default 16384, capped at the model's eval max length).

Expected sanity signals on a healthy 3.28 model: `induction` near-saturated, `control_random` ≈ 0.5 (chance), and the gate marking which tasks are admissible. Only gated-PASS tasks feed the circuit metric; the rest are noise.

### Running circuit discovery

Circuit discovery is implemented as:

- `circuit.py` — model-independent core: builds single-sequence circuit-discovery inputs from the forced-choice tasks (metric = `logit(correct) − logit(distractor)` at the answer position, raw `lm_head`), the faithfulness normalization, the greedy minimal-circuit search with a backward prune, the random-circuit baseline, and the geomean. Torch-free; unit-tested on CPU (`tests/test_circuit.py`).
- `circuit_patcher.py` — model-coupled half: the `tap(name, t)` protocol that `GPT._cd_tap` calls, the mean-activation baseline pass, the clean+grad pass (one backward; nodes retain grad so each attribution score is `(clean − mean) · ∂M/∂node`), and the `measure(kept)` mean-ablation forward. CPU-integration-tested against a synthetic autograd model (`tests/test_patcher.py`).
- Default-off taps in `train_gpt.py` (`GPT._cd_tap`, gated by the class-constant `_cd_enabled` so `torch.compile` folds them out of the training graph) at `attn{i}`, `mlp{i}`, `skip6`, the `embed` grad-root, and the `resid_final` readout. A `TRACK5_CIRCUIT=1` branch in the eval hook runs discovery (with grad, uncompiled) and exits.

```bash
TRACK5_PROBE_CKPT=logs/<run_id>/state_step001390.pt TRACK5_CIRCUIT=1 \
TRACK5_CIRCUIT_TASKS=induction,ioi_copy,successor_days,... \
TRACK5_CD_N=32 TRACK5_CD_TAU=0.8 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Knobs: `TRACK5_CD_N` (examples/task, default 48), `TRACK5_CD_TAU` (faithfulness target, default 0.8), `TRACK5_CIRCUIT_TASKS` (comma list). Results on the anchor are in [Appendix B](#appendix-b--v0-circuit-discovery-results-track-1-anchor).

**Node granularity (v1).** Nodes are the component-level residual writers the fused forward exposes: `attn{i}` (i≠6), `mlp{i}`, and the layer-6 `skip` — 22 ablatable nodes; embeddings/bigram/value-embed/MUDD scaffolding stays always-on. Per-head edges (and the architecture-specific injection nodes as first-class ablatables) require an eager attention reimplementation since FA3 hides head internals — the next refinement, expected to shrink absolute sizes.

**Offline tests** (CPU, validates task construction + packing/span alignment):

```bash
.venv311/bin/python -m records.track_5_interpretability.tests.test_packing
```

## Open decisions

- Statistical dominance thresholds and minimum-improvement margin (need v0 variance data).
- Per-task faithfulness target and exact aggregation constants.
- Final admitted task suite (gated by the anchor's measured capabilities).
- Whether EAP-IG is needed from the start or only after observing saturation artifacts.
- AIS specifics: downstream index set (all-downstream vs one-hop readers), signed vs magnitude influence, the exact hub-penalty weighting, and the high/low-AIS thresholds — all to be set from the v1.5 validation on known reusable vs polysemantic hubs.

## Appendix A — v0 capability-probe results (Track 1 anchor)

First empirical pass of the forced-choice capability probe on the dense Track 1 anchor.

- **Checkpoint:** `logs/9c4d9aab-b531-4158-9575-4ddb001770e7/state_step001390.pt`, val loss **3.2798** (GPT-2-small scale: `num_layers=11, num_heads=6, head_dim=128, model_dim=768`).
- **Probe:** `n=200` examples/task, forced choice scored by mean per-token loss; `margin` = loss(best wrong) − loss(correct) (higher ⇒ more confident-correct). Gates: tier 1 ≥ 0.80, tier 2 ≥ 0.65, control ungated.
- **Reproducibility:** examples are seeded deterministically (`crc32`, process-independent), so the table is stable across runs. Absolute accuracies/margins are mildly runtime-dependent (this run: torch 2.10+cu128); the **PASS/fail gate is the robust quantity to pin**, not raw margins.

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

### What the anchor is good at — and the unifying deficit

The pass/fail split is **mechanistically coherent**, not random. The model is strong wherever a task reduces to **copy / induction / sequence continuation / local syntax / lookup**, and fails wherever it must **select a structurally-correct token over a more *salient* (recent or duplicated) competitor**.

- **Strong (gated PASS):** name copying (`ioi_copy` 1.00), induction (0.90), successors over digits/days/months/letters (0.87–1.00), subject–verb agreement *across* a number attractor (1.00), gender-pronoun resolution (0.98), stored factual knowledge (`factual_recall` 1.00, `greater_than` 0.80), and a lone bracket pair (`bracket_d0` 0.83). Knowledge and agreement are essentially saturated.
- **Fails (excluded by gate):** every "select-over-salience" task — `ioi` (0.45, *prefers the subject*), `ioi_oneshot` (0.11), `verbatim_recall` / key-value binding (0.35, *below chance — copies the most recent value*), and bracket matching at any nesting depth ≥ 1 (0.47–0.585). Arithmetic (`single_digit_addition` 0.47) fails for the separate reason that there is no compute circuit.

Three independent signatures converge on the same missing capability — an **inhibition / variable-binding** mechanism:

1. **IOI ladder.** `ioi_copy` 1.00 → `ioi` 0.45 → `ioi_oneshot` 0.11. Copying is perfect, but the moment a duplicated subject competes, the model follows it; a worked demonstration makes it *worse*, because the model's dominant in-context tool (induction) latches onto the salient repeated token. The missing piece is specifically **S-inhibition**, not name copying or in-context learning.
2. **Verbatim recall (binding).** Below chance (0.35): the model copies the most-recently-stated value instead of binding the queried key — and this *dissociates from induction* (0.90 vs 0.35), so it is a genuinely distinct, harder probe.
3. **Bracket depth curve.** Monotone decay `d0 6.93 → d1 2.98 → d2 1.87 → d3 0.97 → d4 0.18` (margin): it can close a lone pair but cannot reliably bind a closer to the *innermost* opener when an enclosing bracket competes.

**Implication for the track.** A fast/cheap model readily acquires copying, induction, retrieval, and knowledge, but **not the inhibition/binding circuits** that the weight-sparse interpretability literature treats as the canonical clean multi-step circuits (IOI being the flagship). This sharpens the benchmark's central question into a concrete, measurable one: *does spending more compute (or weight sparsity) buy the binding/inhibition circuits this dense anchor lacks?* The failing tasks — IOI family, verbatim binding, deep brackets — are where that signal should first appear, so they are worth keeping as **held-out diagnostics** even though they are excluded from the gated circuit-size metric.

**Gated-PASS suite used for the circuit-size metric:** induction, ioi_copy, successor_{days,months,digits,letters}, bracket_d0, factual_recall, greater_than, pronoun_gender, subject_verb_agreement — a spread spanning induction/copy heads, positional/successor heads, MLP lookup, and agreement.

## Appendix B — v0 circuit-discovery results (Track 1 anchor)

First attribution-patching circuit-discovery pass on the same checkpoint as Appendix A.

- **Method:** node attribution patching over the 22 component-level residual writers (`attn{i}`, `mlp{i}`, `skip6`); mean-ablation baseline computed per task; metric = `logit(correct) − logit(distractor)` (raw `lm_head`); greedy add by `|score|` to measured faithfulness `τ = 0.8`, then backward prune. `n = 32` examples/task. **Faithfulness is measured by real ablation, not the AP estimate.**
- **`M_full`** = mean clean logit difference (task-behavior strength). **`F`** = measured faithfulness of the discovered circuit. **`rand@size`** = mean ± std faithfulness of a random circuit of the same size (the noise floor).

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

**Geometric-mean circuit size = 13.97** over the 11 gated-PASS tasks (`τ = 0.8`).

### Reading the v0 numbers

- **AP ≫ random everywhere.** Every random-circuit floor sits near 0 (range −0.01 to 0.21) while the AP circuits clear 0.8, so the size signal is real, not an artifact of how many nodes are kept.
- **Sizes are large in absolute terms (11–18 of 22)** because the node graph is *component-level*: with one node per attention/MLP layer, most layers genuinely participate, so the metric currently measures "how many components" rather than "which heads". This is the expected coarse-granularity regime; per-head edges (next refinement) should both shrink sizes and spread them, giving the metric more dynamic range. The absolute number is an anchor baseline, not a target — the leaderboard cares about *relative* movement of this geomean as compute/sparsity change.
- **`F > 1` (induction 1.24).** The circuit over-recovers the clean logit difference: some excluded components are *suppressors* whose mean-ablation raises the metric. This is a faithful, expected signature (negative/anti-induction components), reported as-is rather than clipped.
- **Capability strength ≠ circuit size.** `ioi_copy` is both the strongest behavior (`M_full +23`) and the smallest circuit (11) — a clean, concentrated copy mechanism — whereas `greater_than` is a weak behavior (`M_full +3.1`) with the largest, most diffuse circuit (18). Diffuse-and-weak is exactly the profile we'd expect more interpretable training to tighten.

## References

- [Gao et al., "Weight-sparse transformers have interpretable circuits," arXiv:2511.13653 (2025).](https://arxiv.org/pdf/2511.13653) ([code](https://github.com/openai/circuit_sparsity/))
- [Hanna et al., "Have Faith in Faithfulness: ... EAP-IG," arXiv:2403.17806 (2024).](https://arxiv.org/abs/2403.17806)
- [Wang et al., "Interpretability in the Wild: ... IOI circuit in GPT-2 small," arXiv:2211.00593 (2022).](https://arxiv.org/abs/2211.00593)
- [Olsson et al., "In-context Learning and Induction Heads," (2022).](https://arxiv.org/abs/2209.11895)
