# M2 Reverse — Task 2 experiment
# Identical hyperparameters to M1, only dataset differs
out_dir = 'out-m2-reverse'
dataset = 'rocstories_reverse'
wandb_log = False
wandb_run_name = 'M2-reverse'

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

n_layer = 4
n_head = 4
n_embd = 256
block_size = 512
dropout = 0.35
bias = False

batch_size = 32
gradient_accumulation_steps = 2
max_iters = 5000
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 5000
warmup_iters = 200
decay_lr = True

beta1 = 0.9
beta2 = 0.99
weight_decay = 0.2
grad_clip = 1.0
compile = True
