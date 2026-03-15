# ===========================================================================
# M2 — Reverse Causal (Full Training)
# ===========================================================================
# Stories reversed: S5 → S4 → S3 → S2 → S1
# Tests H1: does narrative direction (causal order) matter to an LM?
# Hypothesis: higher perplexity than M1 (causal asymmetry in narrative text)
#
# Prepare data first:
#   python data/rocstories/prepare.py --variant=reverse
#
# Run training:
#   python train.py config/train_m2_reverse.py
#
# CRITICAL: Hyperparameters are IDENTICAL to M1_BEST — only the data changes.
# This ensures a scientifically valid comparison.
# ===========================================================================

# Output
out_dir = 'out-m2-reverse'
wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'M2-reverse-causal'

# Data
dataset = 'rocstories_reverse'  # maps to data/rocstories_reverse/

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
