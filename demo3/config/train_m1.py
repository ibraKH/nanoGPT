# M1 Forward — SUBMISSION MODEL
# Full training config: 5000 iters, block_size=512, dropout=0.3
out_dir = 'out-m1'
dataset = 'rocstories'
wandb_log = False
wandb_run_name = 'M1-forward'

eval_interval = 250
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

n_layer = 6
n_head = 6
n_embd = 384
block_size = 512
dropout = 0.3
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
weight_decay = 0.15
grad_clip = 1.0
compile = True
