# model parallel GPU-GPU
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import os


os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = ""
model_id = "meta-llama/Llama-2-7b-chat-hf"

prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)


tokenizer = AutoTokenizer.from_pretrained(model_id,  cache_dir='/scratch/npattab1/llms/', padding_side='left')
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    cache_dir='/scratch/npattab1/llms/',
    attn_implementation="flash_attention_2",
)

generated_text = []
with torch.inference_mode():
      tokens = tokenizer(prompts,  return_tensors="pt", padding='longest').to('cuda')
      output = model.generate(**tokens, max_new_tokens=1024, do_sample=True, temperature=1.0,
                              # num_beams=5, no_repeat_ngram_size=2
                             )
      output = output[:, tokens["input_ids"].shape[1]:]
      generated_text += tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_text)
