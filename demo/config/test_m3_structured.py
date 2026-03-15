# ===========================================================================
# M3 — Structured Tokens — SMOKE TEST (50 iterations only)
# ===========================================================================
# Purpose: Verify M3 data pipeline (including <S1>...<S5> tokens) works.
#
# Run:
#   python train.py config/test_m3_structured.py
# ===========================================================================

out_dir = 'out-test-m3'
wandb_log = False

dataset = 'rocstories_struct'

max_iters = 50
eval_interval = 25
eval_iters = 10
log_interval = 5
always_save_checkpoint = False

n_layer = 6
n_head = 6
n_embd = 384
block_size = 256
dropout = 0.2
bias = False

batch_size = 64
gradient_accumulation_steps = 1

learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 50
warmup_iters = 5
decay_lr = True

beta1 = 0.9
beta2 = 0.99
weight_decay = 1e-1
grad_clip = 1.0

compile = False
