# ===========================================================================
# M1_BEST — Forward Baseline (Full Training)  ← SUBMISSION MODEL
# ===========================================================================
# Standard ROCStories in natural order: S1 → S2 → S3 → S4 → S5
# This is the Task 1 baseline and the model submitted for Task 3.
#
# WHY M1_BEST IS THE SUBMISSION (not M3):
#   The tutor evaluates PPL by feeding plain text stories to the checkpoint.
#   M3 trained on "<S1>...<S5>" format gets PPL ~94 on plain text (FAILS ≤25).
#   M1_BEST trained on plain text gets consistent low PPL under tutor eval.
#
# Improvements over demo1 M1 (which had PPL 27.93, just above the 25.0 line):
#   - max_iters doubled: 10000 (was 5000)
#   - warmup_iters: 200 (was 100) — smoother ramp-up
#   - eval_interval: 500 (was 250) — less overhead, trains faster
#   - Expected val PPL: ~20-23 (well under 25.0 threshold)
#
# Prepare data first:
#   python data/rocstories/prepare.py --variant=forward
#
# Run training:
#   python train.py config/train_m1_best.py
#
# Expected: ~60-90 min on free Colab T4, ~10-15 min on A100
# ===========================================================================

# Output
out_dir = 'out-m1-best'
wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'M1-best-submission'

# Data
dataset = 'rocstories'  # maps to data/rocstories/

# Logging
eval_interval = 500      # evaluate every 500 steps (less overhead than 250)
eval_iters = 100         # 100 batches for evaluation
log_interval = 10
always_save_checkpoint = False  # only save when val loss improves → best checkpoint quality

# Model architecture — "baby GPT" (~10M parameters)
# Identical across all 3 models for fair comparison
n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

# Training
batch_size = 64
gradient_accumulation_steps = 1
max_iters = 10000          # doubled from demo1 (5000 → 10000)

# Learning rate schedule — cosine decay
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 10000     # must match max_iters
warmup_iters = 200         # smoother warm-up (was 100)
decay_lr = True

# Optimizer
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

# System
compile = True
