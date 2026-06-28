"""Forced-choice capability probes for the interpretability benchmark (Track 5).

Each task produces `ProbeExample`s: a shared context plus a list of candidate
continuations, exactly one of which is correct. This mirrors the binary
next-token framing of Gao et al. (2511.13653) and reuses the same packing /
loss-per-token scoring machinery as `evals/hellaswag.py`.

This module is deliberately torch-free so the task construction can be unit
tested on CPU with only `tiktoken` installed. The model forward pass lives in
`probe_harness.py`.

Design notes
------------
- We build candidates as token-id lists and concatenate them to the context
  token ids directly (no re-tokenization of the join), matching hellaswag. This
  keeps candidate spans exact and avoids BPE-boundary surprises during scoring.
- Every task also exposes a `corrupt_context` where it is natural (a minimal
  pair that flips the correct answer). The capability probe does not need it,
  but the later attribution-patching phase does, so we store it now.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from functools import lru_cache, partial

import tiktoken


@lru_cache(1)
def _enc() -> "tiktoken.Encoding":
    return tiktoken.get_encoding("gpt2")


def _tok(s: str) -> list[int]:
    return _enc().encode_ordinary(s)


def _single_tok(s: str) -> int | None:
    """Return the token id if `s` encodes to exactly one token, else None."""
    ids = _enc().encode_ordinary(s)
    return ids[0] if len(ids) == 1 else None


@dataclass
class ProbeExample:
    """One forced-choice item.

    context: token ids of the shared prefix.
    candidates: list of token-id lists; the model scores each by mean loss.
    label: index into `candidates` of the correct continuation.
    corrupt_context: optional minimal-pair prefix whose correct answer is a
        different candidate index (`corrupt_label`). Used later for patching.
    """

    context: list[int]
    candidates: list[list[int]]
    label: int
    task: str
    corrupt_context: list[int] | None = None
    corrupt_label: int | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Tier 1: induction
# --------------------------------------------------------------------------

# A pool of "ordinary" token ids to sample random sequences from. We avoid very
# low ids (bytes/punctuation) and the reserved EOT id (50256). Induction is
# token-identity based, so the exact pool only needs to be diverse + stable.
_INDUCTION_POOL = list(range(300, 50000))


def induction(n: int, rng: random.Random, block_len: int = 16) -> list[ProbeExample]:
    """Repeated-random-sequence induction.

    A block of distinct random tokens is repeated; given the block followed by a
    prefix of its second occurrence ending at block[k-1], the correct next token
    is block[k] (seen earlier right after block[k-1]). The distractor is another
    in-context block token, so the task specifically rewards prefix-matching
    induction rather than unigram frequency.
    """
    out: list[ProbeExample] = []
    for _ in range(n):
        block = rng.sample(_INDUCTION_POOL, block_len)
        k = rng.randint(1, block_len - 1)
        context = block + block[:k]
        correct = block[k]
        # distractor: a different in-context block token
        j = (k + block_len // 2) % block_len
        if j == k or block[j] == correct:
            j = (k + 1) % block_len
        wrong = block[j]
        cands = [[correct], [wrong]]
        out.append(ProbeExample(context=context, candidates=cands, label=0, task="induction",
                                meta={"block_len": block_len, "k": k}))
    return out


# --------------------------------------------------------------------------
# Tier 1: successor (days / months / digits)
# --------------------------------------------------------------------------

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]


def _successor_cyclic(n: int, rng: random.Random, items: list[str], task: str,
                      run: int = 3) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    m = len(items)
    for _ in range(n):
        start = rng.randrange(m)
        seq = [items[(start + i) % m] for i in range(run)]
        nxt = items[(start + run) % m]
        # distractor: a non-adjacent item (neither the answer nor the last shown)
        bad = items[(start + run + 2) % m]
        context = _tok("".join(" " + w for w in seq))
        correct = _single_tok(" " + nxt)
        wrong = _single_tok(" " + bad)
        if correct is None or wrong is None or correct == wrong:
            continue
        out.append(ProbeExample(context=context, candidates=[[correct], [wrong]],
                                label=0, task=task, meta={"seq": seq, "next": nxt}))
    return out


def successor_days(n: int, rng: random.Random) -> list[ProbeExample]:
    return _successor_cyclic(n, rng, _DAYS, "successor_days")


def successor_months(n: int, rng: random.Random) -> list[ProbeExample]:
    return _successor_cyclic(n, rng, _MONTHS, "successor_months")


def successor_digits(n: int, rng: random.Random, run: int = 3) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    for _ in range(n):
        start = rng.randint(0, 6 - run)  # keep answer <= 9 (single token)
        seq = [start + i for i in range(run)]
        nxt = start + run
        bad = (nxt + 3) % 10
        if bad == nxt:
            bad = (nxt + 4) % 10
        context = _tok("".join(" " + str(d) for d in seq))
        correct = _single_tok(" " + str(nxt))
        wrong = _single_tok(" " + str(bad))
        if correct is None or wrong is None or correct == wrong:
            continue
        out.append(ProbeExample(context=context, candidates=[[correct], [wrong]],
                                label=0, task="successor_digits", meta={"seq": seq, "next": nxt}))
    return out


# --------------------------------------------------------------------------
# Tier 2: IOI (indirect object identification)
# --------------------------------------------------------------------------

_NAMES = ["John", "Mary", "Tom", "Sarah", "David", "Anna", "Paul", "Laura",
          "Mark", "Emma", "Peter", "Julia"]
_IOI_TEMPLATES = [
    "When {A} and {B} went to the store, {S} gave a drink to",
    "After {A} and {B} arrived at the party, {S} handed the keys to",
    "While {A} and {B} were working, {S} passed the report to",
]


def ioi(n: int, rng: random.Random) -> list[ProbeExample]:
    """Indirect Object Identification.

    "When A and B went to the store, A gave a drink to" -> B (the indirect
    object, i.e. the name mentioned once). The minimal-pair corrupt swaps the
    subject S (A<->B), which flips the correct answer.
    """
    out: list[ProbeExample] = []
    # precompute single-token names with leading space
    name_tok = {nm: _single_tok(" " + nm) for nm in _NAMES}
    names = [nm for nm in _NAMES if name_tok[nm] is not None]
    for _ in range(n):
        a, b = rng.sample(names, 2)
        tmpl = rng.choice(_IOI_TEMPLATES)
        # subject = A => answer is B (index 0); corrupt subject = B => answer A
        ctx = _tok(tmpl.format(A=a, B=b, S=a))
        corrupt = _tok(tmpl.format(A=a, B=b, S=b))
        cands = [[name_tok[b]], [name_tok[a]]]
        out.append(ProbeExample(context=ctx, candidates=cands, label=0, task="ioi",
                                corrupt_context=corrupt, corrupt_label=1,
                                meta={"A": a, "B": b}))
    return out


# --------------------------------------------------------------------------
# Tier 2: greater-than (year completion)
# --------------------------------------------------------------------------

_GT_NOUNS = ["war", "siege", "voyage", "reign", "drought", "famine", "expedition"]


def greater_than(n: int, rng: random.Random) -> list[ProbeExample]:
    """Greater-than proxy.

    "The war lasted from the year 17YY to the year 17" -> a two-digit ending
    that must be > YY. Forced choice between a valid (>YY) and invalid (<YY)
    two-digit token. Corrupt swaps in a much smaller YY so the "<" candidate is
    no longer obviously wrong (kept for the patching phase).
    """
    out: list[ProbeExample] = []
    for _ in range(n):
        yy = rng.randint(30, 70)
        zz = rng.randint(yy + 5, 99)   # valid: greater
        ww = rng.randint(1, yy - 5)    # invalid: smaller
        noun = rng.choice(_GT_NOUNS)
        ctx = _tok(f"The {noun} lasted from the year 17{yy:02d} to the year 17")
        zz_tok = _single_tok(f"{zz:02d}")
        ww_tok = _single_tok(f"{ww:02d}")
        if zz_tok is None or ww_tok is None or zz_tok == ww_tok:
            continue
        out.append(ProbeExample(context=ctx, candidates=[[zz_tok], [ww_tok]],
                                label=0, task="greater_than",
                                meta={"yy": yy, "zz": zz, "ww": ww}))
    return out


# --------------------------------------------------------------------------
# Tier 1: alphabet successor
# --------------------------------------------------------------------------

_LETTERS = [chr(ord("a") + i) for i in range(26)]


def successor_letters(n: int, rng: random.Random, run: int = 3) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    for _ in range(n):
        start = rng.randint(0, 26 - run - 1)
        seq = _LETTERS[start:start + run]
        nxt = _LETTERS[start + run]
        bad = _LETTERS[(start + run + 3) % 26]
        context = _tok("".join(" " + c for c in seq))
        correct = _single_tok(" " + nxt)
        wrong = _single_tok(" " + bad)
        if correct is None or wrong is None or correct == wrong:
            continue
        out.append(ProbeExample(context=context, candidates=[[correct], [wrong]],
                                label=0, task="successor_letters", meta={"seq": seq, "next": nxt}))
    return out


# --------------------------------------------------------------------------
# Tier 2: IOI ladder -- copy (L0) and one-shot primed (L2)
#
# `ioi` (standard) fails at chance on the Track 1 anchor; these bracket it so we
# can localise the failure. `ioi_copy` needs only a copy/in-context head (no
# S-inhibition); `ioi_oneshot` prepends a solved example so an induction head can
# carry the pattern. If copy/oneshot pass but standard fails, the missing piece
# is specifically S-inhibition, not name copying.
# --------------------------------------------------------------------------

_COPY_TEMPLATES = [
    "When {A} went to the store, {A} gave a drink to",
    "After {A} arrived at the party, {A} handed the keys to",
    "While {A} was working, {A} passed the report to",
]


def ioi_copy(n: int, rng: random.Random) -> list[ProbeExample]:
    """Single in-context name vs an unseen name. Correct = the seen name."""
    name_tok = {nm: _single_tok(" " + nm) for nm in _NAMES}
    names = [nm for nm in _NAMES if name_tok[nm] is not None]
    out: list[ProbeExample] = []
    for _ in range(n):
        a, b = rng.sample(names, 2)  # a is in-context, b is the unseen distractor
        tmpl = rng.choice(_COPY_TEMPLATES)
        ctx = _tok(tmpl.format(A=a))
        cands = [[name_tok[a]], [name_tok[b]]]
        out.append(ProbeExample(context=ctx, candidates=cands, label=0, task="ioi_copy",
                                meta={"A": a, "B": b}))
    return out


_ONESHOT_PRIME = "When John and Mary met at the cafe, John gave a coffee to Mary. "


def ioi_oneshot(n: int, rng: random.Random) -> list[ProbeExample]:
    """Standard IOI prepended with one solved demonstration (disjoint names)."""
    name_tok = {nm: _single_tok(" " + nm) for nm in _NAMES}
    names = [nm for nm in _NAMES if name_tok[nm] is not None
             and nm not in ("John", "Mary")]
    out: list[ProbeExample] = []
    for _ in range(n):
        a, b = rng.sample(names, 2)
        tmpl = rng.choice(_IOI_TEMPLATES)
        ctx = _tok(_ONESHOT_PRIME + tmpl.format(A=a, B=b, S=a))
        corrupt = _tok(_ONESHOT_PRIME + tmpl.format(A=a, B=b, S=b))
        cands = [[name_tok[b]], [name_tok[a]]]
        out.append(ProbeExample(context=ctx, candidates=cands, label=0, task="ioi_oneshot",
                                corrupt_context=corrupt, corrupt_label=1, meta={"A": a, "B": b}))
    return out


# --------------------------------------------------------------------------
# Tier 2: gender-pronoun resolution
# --------------------------------------------------------------------------

_MALE_NAMES = ["John", "Tom", "David", "Paul", "Mark", "Peter"]
_FEMALE_NAMES = ["Mary", "Sarah", "Anna", "Laura", "Emma", "Julia"]
_PRONOUN_TEMPLATES = [
    "{name} was tired after the long shift, so",
    "When the meeting finally ended, {name} said that",
    "{name} picked up the heavy bag because",
]


def pronoun_gender(n: int, rng: random.Random) -> list[ProbeExample]:
    """Continue a single-subject sentence with the gender-correct pronoun.

    Minimal-pair corrupt swaps in an opposite-gender name, flipping the answer.
    """
    he, she = _single_tok(" he"), _single_tok(" she")
    out: list[ProbeExample] = []
    for _ in range(n):
        male = rng.random() < 0.5
        name = rng.choice(_MALE_NAMES if male else _FEMALE_NAMES)
        other = rng.choice(_FEMALE_NAMES if male else _MALE_NAMES)
        tmpl = rng.choice(_PRONOUN_TEMPLATES)
        ctx = _tok(tmpl.format(name=name))
        corrupt = _tok(tmpl.format(name=other))
        correct_p, wrong_p = (he, she) if male else (she, he)
        out.append(ProbeExample(context=ctx, candidates=[[correct_p], [wrong_p]], label=0,
                                task="pronoun_gender", corrupt_context=corrupt, corrupt_label=1,
                                meta={"name": name, "male": male}))
    return out


# --------------------------------------------------------------------------
# Tier 2: subject-verb number agreement across an attractor noun
# --------------------------------------------------------------------------

# Nouns whose plural is a clean "+s" (avoids "box"->"boxes", "shelf"->"shelves").
_SVA_SUBJ = ["key", "book", "door", "lamp", "chair", "clock", "plate", "card"]
_SVA_ATTR = ["cabinet", "table", "drawer", "office", "window", "garden"]


def subject_verb_agreement(n: int, rng: random.Random) -> list[ProbeExample]:
    """"The key near the cabinets" -> "is" (agree with head, not the attractor).

    The attractor noun always carries the opposite number to the subject, so a
    purely local heuristic is wrong. Corrupt flips the subject number.
    """
    is_t, are_t = _single_tok(" is"), _single_tok(" are")
    out: list[ProbeExample] = []
    for _ in range(n):
        plural = rng.random() < 0.5
        subj, attr = rng.choice(_SVA_SUBJ), rng.choice(_SVA_ATTR)
        subj_w = subj + "s" if plural else subj
        attr_w = attr if plural else attr + "s"
        ctx = _tok(f"The {subj_w} near the {attr_w}")
        c_subj = subj if plural else subj + "s"
        c_attr = attr + "s" if plural else attr
        corrupt = _tok(f"The {c_subj} near the {c_attr}")
        correct, wrong = (are_t, is_t) if plural else (is_t, are_t)
        out.append(ProbeExample(context=ctx, candidates=[[correct], [wrong]], label=0,
                                task="subject_verb_agreement", corrupt_context=corrupt,
                                corrupt_label=1, meta={"plural": plural}))
    return out


# --------------------------------------------------------------------------
# Tier 2: verbatim recall as key-value binding
#
# Two "key holds value" facts are stated, then one key is re-queried. A plain
# induction/recency head copies the most-recent value (the distractor); getting
# it right requires binding the queried key to its own value. This is the part
# of "verbatim recall" that plain induction does NOT cover.
# --------------------------------------------------------------------------

_VR_KEYS = ["red", "blue", "green", "gold", "silver", "black", "white", "brown"]
_VR_TEMPLATES = [
    ("The {k1} box holds {v1}. The {k2} box holds {v2}. The {kq} box holds", "box holds"),
    ("Agent {k1} reported {v1}. Agent {k2} reported {v2}. Agent {kq} reported", "reported"),
]


def verbatim_recall(n: int, rng: random.Random) -> list[ProbeExample]:
    out: list[ProbeExample] = []
    for _ in range(n):
        k1, k2 = rng.sample(_VR_KEYS, 2)
        v1, v2 = rng.sample(range(20, 90), 2)
        v1t, v2t = _single_tok(" " + str(v1)), _single_tok(" " + str(v2))
        if v1t is None or v2t is None or v1t == v2t:
            continue
        tmpl, _ = rng.choice(_VR_TEMPLATES)
        ctx = _tok(tmpl.format(k1=k1, k2=k2, v1=v1, v2=v2, kq=k1))
        corrupt = _tok(tmpl.format(k1=k1, k2=k2, v1=v1, v2=v2, kq=k2))
        out.append(ProbeExample(context=ctx, candidates=[[v1t], [v2t]], label=0,
                                task="verbatim_recall", corrupt_context=corrupt, corrupt_label=1,
                                meta={"k1": k1, "k2": k2, "v1": v1, "v2": v2}))
    return out


# --------------------------------------------------------------------------
# Tier 2: single-digit addition
# --------------------------------------------------------------------------

def single_digit_addition(n: int, rng: random.Random) -> list[ProbeExample]:
    """"{a} + {b} =" -> a+b, distractor is a near-miss sum. Corrupt changes an
    operand so the correct answer becomes the distractor value."""
    out: list[ProbeExample] = []
    for _ in range(n):
        a, b = rng.randint(1, 9), rng.randint(1, 9)
        s = a + b
        delta = rng.choice([-2, -1, 1, 2])
        d = s + delta
        if d < 0 or d == s:
            d = s + 3
        st, dt = _single_tok(" " + str(s)), _single_tok(" " + str(d))
        if st is None or dt is None or st == dt:
            continue
        ctx = _tok(f"{a} + {b} =")
        b2 = b + (d - s)  # operand that would make the answer == distractor
        corrupt = corrupt_label = None
        if 1 <= b2 <= 9:
            corrupt = _tok(f"{a} + {b2} =")
            corrupt_label = 1
        out.append(ProbeExample(context=ctx, candidates=[[st], [dt]], label=0,
                                task="single_digit_addition", corrupt_context=corrupt,
                                corrupt_label=corrupt_label, meta={"a": a, "b": b, "sum": s}))
    return out


# --------------------------------------------------------------------------
# Tier 2: factual recall (capital cities) -- probes stored MLP knowledge
# --------------------------------------------------------------------------

_CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Russia": "Moscow",
    "Egypt": "Cairo", "Greece": "Athens", "Spain": "Madrid", "Germany": "Berlin",
    "England": "London", "China": "Beijing", "Peru": "Lima", "Iran": "Tehran",
    "Cuba": "Havana", "Austria": "Vienna", "Norway": "Oslo", "Poland": "Warsaw",
}


def factual_recall(n: int, rng: random.Random) -> list[ProbeExample]:
    """"The capital of {country} is" -> {capital}, distractor is another capital.
    Corrupt swaps the country so the correct answer becomes the distractor."""
    pairs = [(c, cap) for c, cap in _CAPITALS.items() if _single_tok(" " + cap) is not None]
    out: list[ProbeExample] = []
    for _ in range(n):
        (c1, cap1), (c2, cap2) = rng.sample(pairs, 2)
        a, d = _single_tok(" " + cap1), _single_tok(" " + cap2)
        if a == d:
            continue
        ctx = _tok(f"The capital of {c1} is")
        corrupt = _tok(f"The capital of {c2} is")
        out.append(ProbeExample(context=ctx, candidates=[[a], [d]], label=0,
                                task="factual_recall", corrupt_context=corrupt, corrupt_label=1,
                                meta={"country": c1, "capital": cap1}))
    return out


# --------------------------------------------------------------------------
# Tier-laddered: bracket matching at increasing nesting depth
#
# Space-separated openers are tokenised one-per-token; the correct next token is
# the closer of the INNERMOST (most recent) opener. `depth` = number of enclosing
# brackets around it (depth 0 = a lone pair). The distractor at depth>=1 is the
# closer of the immediately-enclosing opener (an off-by-one-depth wrong match),
# so accuracy degrading with depth isolates how deep the stack tracking holds.
# --------------------------------------------------------------------------

_BRK = [("(", ")"), ("[", "]"), ("{", "}")]


def _bracket_match(n: int, rng: random.Random, depth: int) -> list[ProbeExample]:
    closers = [c for _, c in _BRK]
    close_of = {o: c for o, c in _BRK}
    out: list[ProbeExample] = []
    for _ in range(n):
        opens = [rng.choice(_BRK) for _ in range(depth + 1)]
        inner_c = opens[-1][1]
        if depth >= 1 and opens[-2][1] != inner_c:
            wrong_c = opens[-2][1]  # the enclosing closer: tempting off-by-one
        else:
            wrong_c = rng.choice([c for c in closers if c != inner_c])
        correct, wrong = _single_tok(" " + inner_c), _single_tok(" " + wrong_c)
        if correct is None or wrong is None or correct == wrong:
            continue
        ctx = _tok(" ".join(o for o, _ in opens))
        wrong_o = {c: o for o, c in _BRK}[wrong_c]
        corrupt_opens = opens[:-1] + [(wrong_o, wrong_c)]
        corrupt = _tok(" ".join(o for o, _ in corrupt_opens))
        out.append(ProbeExample(context=ctx, candidates=[[correct], [wrong]], label=0,
                                task=f"bracket_d{depth}", corrupt_context=corrupt, corrupt_label=1,
                                meta={"depth": depth, "opens": [o for o, _ in opens]}))
    return out


# --------------------------------------------------------------------------
# Control: structureless task to calibrate the chance/uninterpretable floor
# --------------------------------------------------------------------------

def control_random(n: int, rng: random.Random, ctx_len: int = 16) -> list[ProbeExample]:
    """Random context, two random candidate tokens, random label.

    The model should sit at chance (~50%) here. Used to calibrate the
    capability gate and, later, the random/uninterpretable floor for the
    circuit metric.
    """
    out: list[ProbeExample] = []
    for _ in range(n):
        context = [rng.choice(_INDUCTION_POOL) for _ in range(ctx_len)]
        c0, c1 = rng.sample(_INDUCTION_POOL, 2)
        label = rng.randint(0, 1)
        out.append(ProbeExample(context=context, candidates=[[c0], [c1]],
                                label=label, task="control_random"))
    return out


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# name -> (tier, generator)
TASK_REGISTRY = {
    "induction":             (1, induction),
    "successor_days":        (1, successor_days),
    "successor_months":      (1, successor_months),
    "successor_digits":      (1, successor_digits),
    "successor_letters":     (1, successor_letters),
    "ioi_copy":              (1, ioi_copy),
    "ioi":                   (2, ioi),
    "ioi_oneshot":           (2, ioi_oneshot),
    "pronoun_gender":        (2, pronoun_gender),
    "subject_verb_agreement": (2, subject_verb_agreement),
    "greater_than":          (2, greater_than),
    "verbatim_recall":       (2, verbatim_recall),
    "single_digit_addition": (2, single_digit_addition),
    "factual_recall":        (2, factual_recall),
    "bracket_d0":            (1, partial(_bracket_match, depth=0)),
    "bracket_d1":            (1, partial(_bracket_match, depth=1)),
    "bracket_d2":            (2, partial(_bracket_match, depth=2)),
    "bracket_d3":            (2, partial(_bracket_match, depth=3)),
    "bracket_d4":            (2, partial(_bracket_match, depth=4)),
    "control_random":        (0, control_random),
}


def build_all(n_per_task: int = 200, seed: int = 0,
              tasks: list[str] | None = None) -> dict[str, list[ProbeExample]]:
    """Generate examples for each task with a deterministic per-task seed."""
    names = tasks if tasks is not None else list(TASK_REGISTRY)
    result: dict[str, list[ProbeExample]] = {}
    for name in names:
        _tier, gen = TASK_REGISTRY[name]
        # Deterministic per-task seed. `hash()` on str/tuple is salted per process
        # (PYTHONHASHSEED), which would make the benchmark non-reproducible across
        # runs; crc32 of a stable byte string is process-independent.
        rng = random.Random(zlib.crc32(f"{seed}:{name}".encode()))
        result[name] = gen(n_per_task, rng)
    return result


def tier_of(name: str) -> int:
    return TASK_REGISTRY[name][0]
