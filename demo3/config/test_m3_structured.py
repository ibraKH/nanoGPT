# M3 Structured — 50 iter smoke test
out_dir = 'out-test-m3'
dataset = 'rocstories_struct'
wandb_log = False

max_iters = 50
eval_interval = 25
eval_iters = 10
log_interval = 5
always_save_checkpoint = False
lr_decay_iters = 50
warmup_iters = 5
compile = False

n_layer = 4
n_head = 4
n_embd = 256
block_size = 512
dropout = 0.35
bias = False

batch_size = 32
gradient_accumulation_steps = 2
learning_rate = 1e-3
min_lr = 1e-4
decay_lr = True

beta1 = 0.9
beta2 = 0.99
weight_decay = 0.2
grad_clip = 1.0
