"""
This script was adapted from https://github.com/karpathy/nanoGPT/blob/master/train.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from collections import defaultdict
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
# put this chunck outside of training function

class Trainer():
    def __init__(self,config,model,tokenizer,dataloader):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.device_type = 'cuda' if 'cuda' in self.config.device else 'cpu' # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.config.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        self.elapsed_epoches = 0
       
        pass

    def get_batch(self, split):
        # this function should just be a wrapper for dataloader.get_batch()
        x,y = self.dataloader.get_batch(split,block_size=self.config.block_size,
                                        batch_size=self.config.batch_size)
        x,y = torch.from_numpy(x), torch.from_numpy(y)

        if self.device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.config.device, non_blocking=True), y.pin_memory().to(self.config.device, non_blocking=True)
        else:
            x, y = x.to(self.config.device), y.to(self.config.device)
        return x, y

    def get_lr(self,it):
        # learning rate decay scheduler (cosine with warmup)
        # but we want to use epoch??
        # 1) linear warmup for warmup_iters steps
        if it < self.config.warmup_iters:
            return self.config.learning_rate * it / self.config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.config.lr_decay_iters:
            return self.config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
     
    @torch.no_grad()
    def estimate_loss(self):
        # helps estimate an arbitrarily accurate loss over either split using many batches
        out = {}
        self.model.eval()
        for split in [self.config.train_or_dev, 'test']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    response = self.model(input_ids=X, labels=Y,return_dict=True)
                    logits, loss = response.logits, response.loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        # various inits, derived attributes, I/O setup
        ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
        local_device = self.config.device
        if ddp:
            init_process_group(backend=self.config.backend)
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            local_device = f'cuda:{ddp_local_rank}'
            torch.cuda.set_device(local_device)
            master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
            seed_offset = ddp_rank # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.config.gradient_accumulation_steps % ddp_world_size == 0
            self.config.gradient_accumulation_steps //= ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            master_process = True
            seed_offset = 0
            ddp_world_size = 1
        tokens_per_iter = self.config.gradient_accumulation_steps * ddp_world_size * self.config.batch_size * self.config.block_size
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

        if master_process:
            os.makedirs(self.config.out_dir, exist_ok=True)
        torch.manual_seed(1337 + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        
        # move self.model to device
        self.model.to(local_device)

        # starting from scratch
        iter_num = 0
        best_val_loss = 1e9

        # initialize a GradScaler. If enabled=False scaler is a no-op
        #scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
        scaler = torch.amp.GradScaler('cuda')# dtype=torch.float16

        # optimizer
        optimizer = self.model.configure_optimizers(self.config.weight_decay, 
        self.config.learning_rate,
        (self.config.beta1, self.config.beta2),
        self.device_type)
        #exit()
        if self.config.init_from == 'resume':
            # do this in future
            pass
            #optimizer.load_state_dict(checkpoint['optimizer'])
        #checkpoint = None # free up memory

        # compile the model
        if self.config.compile:
            print("compiling the model... (takes a ~minute)")
            unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0

        # wrap model into DDP container
        if ddp:
            self.model = DDP(self.model, device_ids=[ddp_local_rank])

        # logging
        if self.config.wandb_log and master_process:
            import wandb
            wandb.init(project=self.config.wandb_project, name=self.config.wandb_run_name,
            config=self.config, reinit=True)

        # training loop
        X, Y = self.get_batch(self.config.train_or_dev) # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0 # number of iterations in the lifetime of this process
        raw_model = self.model.module if ddp else self.model # unwrap DDP container if needed
        running_mfu = -1.0
        print('Training Started ...')
        while True:

            # determine and set the learning rate for this iteration
            lr = self.get_lr(iter_num) if self.config.decay_lr else self.config.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.config.eval_interval == 0 and master_process:
                losses = self.estimate_loss()
                print(f"step {iter_num}: train loss {losses[self.config.train_or_dev]:.4f}, val loss {losses['test']:.4f}")
            
                if self.config.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses[self.config.train_or_dev],
                        "val/loss": losses['test'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                       
                    })
                if losses['test'] < best_val_loss or self.config.always_save_checkpoint:
                    best_val_loss = losses['test']
                    if iter_num > 0:
                        # use huggingface compatible function
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': self.config,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': raw_model.config,
                        }
                        raw_model.save_pretrained(self.config.out_dir)#,safe_serialization=False
                        print(f"saving checkpoint to {self.config.out_dir}")
                        #torch.save(checkpoint, os.path.join(self.config.out_dir, 'ckpt.pt'))
            
            if iter_num == 0 and self.config.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.config.gradient_accumulation_steps):
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    self.model.require_backward_grad_sync = (micro_step == self.config.gradient_accumulation_steps - 1)
                with self.ctx:
                    response = self.model(input_ids=X, labels=Y,return_dict=True)
                    logits, loss = response.logits, response.loss
                    loss = loss / self.config.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                X, Y = self.get_batch(self.config.train_or_dev)
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()
            # clip the gradient
            if self.config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            # step the optimizer and scaler if training in fp16
            
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.config.log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.config.gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(self.config.batch_size * self.config.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
              
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.config.max_iters:
                break

        if ddp:
            destroy_process_group()
        
     
