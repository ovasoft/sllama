from trainer import Trainer
from data_loader import BaseDataLoader
from trainer_config import TrainerConfig
from src.modeling_sllama import SLLamaForCausalLM, SLLamaForSequenceClassification
from src.configuration_sllama import SLLamaConfig
import sys
from transformers import LlamaTokenizer
from utils import load_config

cfg = load_config()

block_size = 256
tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
model_config = SLLamaConfig(attn_reduction_type='pwa')
train_config = TrainerConfig()
model = SLLamaForCausalLM(model_config)
dataloader = BaseDataLoader('10M',eot_token_id=tokenizer.eos_token_id)
#print(dataloader.get_batch(split='train',total_len=block_size))

print(train_config.out_dir)
#raise ValueError("Debug stop")
trainer = Trainer(config=train_config,model=model,tokenizer=tokenizer,dataloader=dataloader)
trainer.train()
#model.save_pretrained(large_train_config.out_dir) # ,safe_serialization=False
tokenizer.save_pretrained(train_config.out_dir)
