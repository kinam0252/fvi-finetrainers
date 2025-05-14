from diffusers import LTXPipeline
import torch
from torch.profiler import profile, ProfilerActivity

pipe = LTXPipeline.from_pretrained("/home/nas4_user/kinamkim/checkpoint/ltxvideo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # if memory is limited
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = "A cat running in the garden"

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             with_flops=True,
             with_stack=True) as prof:
    _ = pipe(prompt, num_frames=49, height=480 , width=480)  # Keep this short to avoid OOM

print(prof.key_averages().table(sort_by="flops", row_limit=10))
