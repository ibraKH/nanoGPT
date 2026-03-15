# ===========================================================================
# M1 — Forward Baseline (Full Training)
# ===========================================================================
# Standard ROCStories in natural order: S1 → S2 → S3 → S4 → S5
# This is the Task 1 baseline model.
#
# Prepare data first:
#   python data/rocstories/prepare.py --variant=forward
#
# Run training:
#   python train.py config/train_m1_forward.py
#
# Expected: ~30-60 min on a free T4 GPU (Colab), ~5 min on A100
# Expected val PPL: 30-60 range after 5000 steps
# ===========================================================================

# Output
out_dir = 'out-m1-forward'
wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'M1-forward-baseline'

# Data
dataset = 'rocstories'  # maps to data/rocstories/

# Logging
eval_interval = 250      # evaluate every 250 steps
eval_iters = 100         # use 100 batches for evaluation (fast but accurate enough)
log_interval = 10        # print loss every 10 steps
always_save_checkpoint = False  # only save when val loss improves

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
max_iters = 5000

# Learning rate schedule — cosine decay
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 5000
warmup_iters = 100
decay_lr = True

# Optimizer
beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

# System — set compile=False if you get errors on older PyTorch
compile = True
