# $ torchrun --nproc-per-node 4 pippy_llama.py
import sys

sys.path.append("temp/PiPPy")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pippy import annotate_split_points, PipelineStage, pipeline, SplitPoint
from pippy.PipelineSchedule import PipelineScheduleGPipe

os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = "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"


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

# sdpa implementation which is the default torch>2.1.2 fails with the tracing + attention mask kwarg
# with attn_implementation="eager" mode, the forward is very slow for some reason
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    cache_dir='/scratch/npattab1/llms/',
)

model.to(device).eval()

# Input configs
# Create example inputs for the model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir='/scratch/npattab1/llms/',)

prompts = (
    "How do you", "I like to", "Can I help", "You need to",
    "The weather is", "I found a", "What is your", "You are so",
)  # bs = 8

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
example_inputs = {"inputs": inputs["input_ids"]}

# Cut model by equal number of layers per rank
layers_per_rank = model.config.num_hidden_layers // world_size
for i in range(1, world_size):
    annotate_split_points(model,
        {f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING})

# Create a pipeline representation from the model
llama_pipe = pipeline(model, world_size, example_args=inputs["input_ids"])

# Create pipeline stage for each rank
torch.distributed.init_process_group(rank=rank, world_size=world_size)
stage = PipelineStage(llama_pipe, rank, device=device)

schedule = PipelineScheduleGPipe(stage, 1)

# Run
if rank == 0:
    args = inputs["input_ids"]
else:
    args = None

output = schedule.step(args)
                    
if output is not None:
    next_token_logits = output[0][:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1)
    print(tokenizer.batch_decode(next_token))
