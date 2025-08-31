from datasets import load_dataset,DatasetDict
from main import original_model
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
import transformers
import time
from tqdm import tqdm
from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from functools import partial
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
output_dir = f'C:/Users/felix/OneDrive/Desktop/Phi-4-Trained-{str(int(time.time()))}'
torch.cuda.empty_cache()

import os
#disable Weights and Bisases
os.environ['WANDB_DISABLED'] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


ds = "HumanLLMs/Human-Like-DPO-Dataset"
dataset = load_dataset(ds)
train_test = dataset['train'].train_test_split(test_size=0.1)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model_name='microsoft/phi-2'
device_map  = {"": 0}
original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map=device_map,
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True
                                                      )

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,padding_side="left", add_eos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


from transformers import set_seed
seed = 42
set_seed(seed)

test_cases = 1
output_samples = []
output_samples1 = []
for i in range(3001, test_cases+3001):
    prompt = dataset['train'][i]['prompt']
    chosen = dataset['train'][i]['chosen']
    formatted_prompt = f"""Give a human like answer to the Prompt: {prompt} Answer:"""


# Tokenize input
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(original_model.device)

# Generate text
    output_ids = original_model.generate(input_ids, max_length=300)

# Decode output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Extract model's response after "Output:"
    output = output_text.split("Answer:")[-1].strip()


    output_samples.append(output)


    ft_model = PeftModel.from_pretrained(original_model, "C:/Users/felix/OneDrive/Desktop/Phi-4-Trained-1742979083/checkpoint-3000",torch_dtype=torch.float16,is_trainable=False)
    input_ids1 = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(ft_model.device)
    output_ids1 = ft_model.generate(input_ids, max_length=300)
    output_text1 = tokenizer.decode(output_ids1[0], skip_special_tokens=True)
    output1 = output_text1.split("Answer:")[-1].strip()
    output_samples1.append(output1)
    dash_line = '-'.join('' for x in range(100))
    print(dash_line)
    print(f'INPUT PROMPT:\n{formatted_prompt}')
    print(dash_line)
    print(f'MODEL GENERATION - ZERO SHOT:\n{output}')
    print(dash_line)
    print(f'PEFT MODEL:\n{output1}')
