# M4 LLaMA — Task 2 novelty experiment
# Same forward data as M1 (isolates architecture effect)
# Same hyperparameters as M1
out_dir = 'out-m4-llama'
dataset = 'rocstories'
wandb_log = False
wandb_run_name = 'M4-llama'

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
