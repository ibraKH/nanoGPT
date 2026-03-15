# ===========================================================================
# M3 — Structured Tokens (Full Training)
# ===========================================================================
# Stories with explicit sentence markers: <S1> S1 <S2> S2 ... <S5> S5
# Tests H2: does explicit positional structure improve narrative coherence?
# Hypothesis: lower perplexity than M1 (structure reduces model uncertainty)
#
# IMPORTANT — DO NOT SUBMIT THIS CHECKPOINT:
#   M3 is trained on structured text. When the tutor feeds it plain text,
#   PPL spikes to ~94 (FAILS the ≤25.0 threshold). M3 is for Task 2
#   write-up only. Submit M1_BEST (out-m1-best/ckpt.pt) for Task 3.
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
# CRITICAL: Hyperparameters are IDENTICAL to M1_BEST — only the data changes.
# ===========================================================================

# Output
out_dir = 'out-m3-structured'
wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'M3-structured-tokens'

# Data
dataset = 'rocstories_struct'  # maps to data/rocstories_struct/

# Logging — IDENTICAL to M1_BEST
eval_interval = 500
eval_iters = 100
log_interval = 10
always_save_checkpoint = False

# Model architecture — IDENTICAL to M1_BEST
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

# Training — IDENTICAL to M1_BEST
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 10000

# Learning rate schedule — IDENTICAL to M1_BEST
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 10000
warmup_iters = 200
decay_lr = True

# Optimizer — IDENTICAL to M1_BEST
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

# System
compile = True
