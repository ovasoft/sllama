
from datasets import Dataset,DatasetDict
import os, shutil
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
from pathlib import Path
from functools import lru_cache

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[0] / "config.yaml"


@lru_cache()
def load_config():
    try:
        with open(CONFIG_PATH, "r") as config_file:
            return yaml.safe_load(config_file) or {}
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}") from exc


def _get_config_value(section, key):
    config = load_config()
    try:
        section_values = config[section]
    except KeyError as exc:
        raise KeyError(f"Missing '{section}' section in configuration.") from exc

    try:
        return section_values[key]
    except KeyError as exc:
        raise KeyError(f"Missing '{key}' in '{section}' configuration.") from exc


data_path = _get_config_value("babylm", "data_path")
data_forms = _get_config_value('babylm','data_forms')
data_splits = _get_config_value('babylm','data_splits')
data_sizes =  _get_config_value('babylm','data_sizes')
tokenized_data_path = _get_config_value('outputs','tokenized_data')


# lower abstraction, don't call directly
def load_baby_dataset_split_from_text(size,split,form,tokenizer):
    #form = data_forms[0]
    def tokenize(example):
        full = tokenizer(example['text'])
        example['input_ids'] = full['input_ids']
        example['num_tokens'] = len(full['input_ids'])  
        return example

    fpath = f'{split}/{form}.{split}' if split != 'train' else f'{split}_{size}/{form}.{split}'
    with open(os.path.join(data_path,fpath),'r') as f:
        dataset = Dataset.from_dict({'text':list(f.readlines())})
        dataset = dataset.map(tokenize,desc=f'Tokenizing {size} {split} {form}')
        dataset = dataset.remove_columns(['text'])
        return dataset



def create_memmaps(size,tokenizer):
    for split in data_splits:
        for form in data_forms:
            ds = load_baby_dataset_split_from_text(size,split,form,tokenizer)
            if split == 'train':
                tmap = {'100M':'train_100M','10M':'train_10M'}
                sp = tmap[size]
            else:
                sp = split
            dtpath = os.path.join(tokenized_data_path,f'{sp}')
            os.makedirs(dtpath,exist_ok=True)
            id_filename = os.path.join(dtpath,f'{form}.bin')
            # create memmap
            arr_len = sum(ds['num_tokens'])
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            id_arr = np.memmap(id_filename, dtype=dtype, mode='w+', shape=(arr_len,))
            idx = 0
            for row in tqdm(ds,desc=f'Creating memmap for {size} {split} {form}'):
                length = len(row['input_ids'])
                id_arr[idx:idx+length] = row['input_ids']
                idx += length
            id_arr.flush()
                

# utility function to load dataset callable from outside the script
def load_data_memmap(size,form,split):
    sp = split if split != 'train' else f'train_{size}'
    id_filename = os.path.join(tokenized_data_path, f'{sp}/{form}.bin')
    id_bucket = np.memmap(id_filename, dtype=np.uint16, mode='r')
    return id_bucket




def create_or_get_path(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


if __name__ == '__main__':
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
    create_memmaps('10M',tokenizer)



