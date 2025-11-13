from dataclasses import dataclass
from utils import load_config

cfg = load_config()
trainer_cfg = cfg.get('training',{})

@dataclass
class TrainerConfig:
    eval_interval:int = trainer_cfg.get('eval_interval',500)
    log_interval:int = trainer_cfg.get('log_interval',10)
    eval_iters:int = trainer_cfg.get('eval_iters',150)
    eval_only:bool = trainer_cfg.get('eval_only',False) # if True, script exits right after the first eval
    always_save_checkpoint:bool = trainer_cfg.get('always_save_checkpoint',False) # if True, always save a checkpoint after each eval
    init_from:str = trainer_cfg.get('init_from','scratch')  # 'scratch' or 'resume' or 'gpt2*'
    wandb_log:bool = trainer_cfg.get('wandb_log',True)  # disabled by default
    wandb_project:str = trainer_cfg.get('wandb_project', 'sllama') 
    wandb_run_name:str = trainer_cfg.get('wandb_run_name', 'sllama')  # 'run' + str(time.time())
    gradient_accumulation_steps:int = trainer_cfg.get('gradient_accumulation_steps',2) # used to simulate larger batch sizes
    batch_size:int = trainer_cfg.get('batch_size', 128) # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size:int = trainer_cfg.get('block_size', 256)
    dropout:float = trainer_cfg.get('dropout', 0.1) # for pretraining 0 is good, for finetuning try 0.1+
    bias:bool = trainer_cfg.get('bias', False) # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate:float = trainer_cfg.get('learning_rate', 4e-4) # max learning rate
    max_iters:int = trainer_cfg.get('max_iters', 3000) # total number of training iterations
    weight_decay:float = trainer_cfg.get('weight_decay', 0.0)
    layer_sharing: bool = trainer_cfg.get('layer_sharing', False)
    beta1:float = trainer_cfg.get('beta1', 0.9)
    beta2:float = trainer_cfg.get('beta2', 0.95)
    grad_clip:float = trainer_cfg.get('grad_clip', 1.0) # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr:bool = trainer_cfg.get('decay_lr', True) # whether to decay the learning rate
    warmup_iters:int = trainer_cfg.get('warmup_iters', 200) # how many steps to warm up for
    lr_decay_iters:int = trainer_cfg.get('lr_decay_iters', 5000) # should be ~= max_iters per Chinchilla
    min_lr:float = trainer_cfg.get('min_lr', 4e-5) # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # DDP settings
    backend:str = trainer_cfg.get('backend', 'nccl') # 'nccl', 'gloo', etc.
    # system
    device:str = trainer_cfg.get('device', 'cuda') # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype:str = trainer_cfg.get('dtype', 'bfloat16')
    compile:bool = trainer_cfg.get('compile', True) # use PyTorch 2.0 to compile the model to be faster
    save_weights:bool = trainer_cfg.get('save_weights', False)
    train_or_dev:str = trainer_cfg.get('train_or_dev', 'train')
    _out_dir: str = 'sllama_main'
    base_dir = cfg.get('outputs', {}).get('pretrained_models', 'sllama_main')

    @property
    def out_dir(self) -> str:
        return f"{self.base_dir}/{self._out_dir}"# self._out_dir
    
    @out_dir.setter
    def out_dir(self, value: str):
        self._out_dir = f"{self.base_dir}/{value}"



