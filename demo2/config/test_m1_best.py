# ===========================================================================
# M1_BEST — SMOKE TEST (50 iterations only)
# ===========================================================================
# Purpose: Verify the pipeline works end-to-end before full training.
# Runs only 50 iterations — fast (~30 seconds on GPU).
#
# Run:
#   python train.py config/test_m1_best.py
#
# If this completes without errors, the full training will work.
# ===========================================================================

out_dir = 'out-test-m1'
wandb_log = False

dataset = 'rocstories'

max_iters = 50
eval_interval = 25
eval_iters = 10
log_interval = 5
always_save_checkpoint = False

# Same model architecture as full training
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

# Disable torch.compile for faster startup in test mode
compile = False
