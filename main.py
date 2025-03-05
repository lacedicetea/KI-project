from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np

#import os
#disable Weights and Bisases
#os.environ['WANDB_DISABLED'] = 'true
from huggingface_hub import interpreter_login

interpreter_login("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")


ds = "SetFit/emotion"
dataset = load_dataset(ds)
print(dataset)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model_name='mistralai/Mistral-7B-Instruct-v0.3'
device_map  = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left", add_eos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


from transformers import set_seed
seed = 42
set_seed(seed)

index = 10

prompt = dataset['test'][index]['text']
label = dataset['test'][index]['label']
label_text = dataset['test'][index]['label_text']

formatted_prompt = f"Instruct: formulate the text with the given emotion.\n{label}\n{prompt}\nOutput:\n"
res = tokenizer.generate(original_model,formatted_prompt,100,)
#print(res[0])
output = res[0].split('Output:\n')[1]

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{formatted_prompt}')
print(dash_line)
print(f'BASELINE HUMAN formulation:\n{label_text}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')