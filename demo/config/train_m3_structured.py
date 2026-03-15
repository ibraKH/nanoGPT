# ===========================================================================
# M3 — Structured Tokens (Full Training)   ← BEST MODEL / COMPETITION ENTRY
# ===========================================================================
# Stories with explicit sentence markers: <S1> S1 <S2> S2 ... <S5> S5
# Tests H2: does explicit positional structure improve narrative coherence?
# Hypothesis: lower perplexity than M1 (structure reduces uncertainty)
#
# Bonus capability — guided story completion at inference:
#   python sample.py --out_dir=out-m3-structured \
#          --start="<S1> Mary went to the park. <S2>"
#
# Prepare data first:
#   python data/rocstories/prepare.py --variant=structured
#
# Run training:
#   python train.py config/train_m3_structured.py
#
# CRITICAL: Hyperparameters are IDENTICAL to M1 — only the data changes.
# This is your Task 3 submission checkpoint.
# ===========================================================================

# Output
out_dir = 'out-m3-structured'
wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'M3-structured-tokens'

# Data
dataset = 'rocstories_struct'  # maps to data/rocstories_struct/

# Logging
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves

# Model architecture — IDENTICAL to M1
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

# Training — IDENTICAL to M1
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 5000

# Learning rate schedule — IDENTICAL to M1
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 5000
warmup_iters = 100
decay_lr = True

# Optimizer — IDENTICAL to M1
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

# System
compile = True
