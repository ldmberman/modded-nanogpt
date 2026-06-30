from __future__ import annotations

import torch

from .model import GPTConfig, GPT2_VOCAB_SIZE
from .training import Hyperparameters


SMALL_CONFIG = GPTConfig(vocab_size=GPT2_VOCAB_SIZE, num_layers=12, model_dim=768,
                         head_dim=64, mlp_ratio=4)
BIG_CONFIG = GPTConfig(vocab_size=GPT2_VOCAB_SIZE, num_layers=15, model_dim=960,
                       head_dim=64, mlp_ratio=4)


HPARAMS = Hyperparameters()

# Fixed schedule horizons, one per model size, not tunable. Each is sized to a
# realistic budget for that model: halfway in loss between its compute-optimal
# point (~20 tokens/param) and its capacity floor (see utils/chinchilla_budget.py).
# Each schedule cools to that model's realistic target at its endpoint. The small
# budget schedules the L1 checkpoint run; the big budget schedules the from-scratch
# reference and every warm-start and cools to the big realistic target, which sits
# below L2. L2 is therefore a waypoint above the big cooled endpoint, so every big
# run passes L2 on the way down, generally during the cooldown phase rather than at
# peak LR. Comparability comes from every run of a given size sharing this identical
# fixed schedule, not from reading all runs at a common learning rate.
SMALL_BUDGET_STEPS = 62_000   # ~16.3B tokens
BIG_BUDGET_STEPS = 100_000    # ~26.3B tokens

VAL_TOKENS = 10_485_760

# FineWeb10B shards (download with `python data/cached_fineweb10B.py N`).
TRAIN_GLOB = "data/fineweb10B/fineweb_train_*.bin"
VAL_GLOB = "data/fineweb10B/fineweb_val_*.bin"

L1: float | None = 3.25
L2: float | None = 3.18

AMP_DTYPE: torch.dtype | None = torch.bfloat16
COMPILE = True
