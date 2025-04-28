import torch
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video generation validation script")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ltxvideo", "cogvideo"],
        required=False,
        default="cogvideo",
        help="Type of model to use for validation"
    )
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        required=False,
        default=None,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=False,
        default="/home/nas4_user/kinamkim/video-in-context-lora/dataset/P2V/dataset",
        help="Path to dataset to use for validation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="Width of the video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the video"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=30,
        help="Number of videos to generate"
    )
    parser.add_argument(
        "--apply_target_noise_only",
        type=str,
        default="front",
        # required=True,
        help="Apply noise only to target frame"
    )
    return parser.parse_args()


args = parse_args()

@torch.no_grad()
def process_video(pipe, video_path, dtype, generator, height, width, apply_target_noise_only):
    if apply_target_noise_only == None:
        return None
    from diffusers.utils import load_video
    from diffusers.pipelines.cogvideo.pipeline_cogvideox_video2video import retrieve_latents
    from diffusers.utils.torch_utils import randn_tensor
    video = load_video(video_path)
    video = pipe.video_processor.preprocess_video(video, height=height, width=width)
    video = video.to("cuda", dtype=dtype)
    
    video_latents = retrieve_latents(pipe.vae.encode(video))
    init_latents = video_latents.to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
    init_latents = pipe.vae_scaling_factor_image * init_latents
    init_latents = init_latents.to(pipe.device)

    noise = randn_tensor(init_latents.shape, generator=generator, device=pipe.device, dtype=dtype)
    if apply_target_noise_only == "back":
        init_latents[:, :-1] = noise[:, :-1]
    elif apply_target_noise_only == "front":
        init_latents[:, 1:] = noise[:, 1:]
    elif apply_target_noise_only == "front-long":
        init_latents[:, 6:] = noise[:, 6:]
    elif apply_target_noise_only == "front-last-long":
        init_latents[:, 6:-1] = noise[:, 6:-1]
    elif apply_target_noise_only == "front-last-long-long":
        init_latents[:, 5:-3] = noise[:, 5:-3]
    else:
        raise ValueError(f"apply_target_noise_only must be either 'back' or 'front', but got {apply_target_noise_only}")
    init_latents = init_latents.to(pipe.device)
    return init_latents

@torch.no_grad()
def retrieve_video(pipe, init_latents,):
    init_latents = init_latents.to("cuda")
    video = pipe.decode_latents(init_latents)
    video = pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]
    return video

if __name__ == "__main__":
    with torch.no_grad():
        if args.model_type == "ltxvideo":
            from diffusers import LTXPipeline
            from diffusers.utils import export_to_video
            pipe = LTXPipeline.from_pretrained(
                "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
            ).to("cuda")
            pipe.load_lora_weights(args.lora_weight_path, adapter_name="ltxv-lora")
            pipe.set_adapters(["ltxv-lora"], [0.75])

        elif args.model_type == "cogvideo":
            from pipeline import CogVideoXPipeline
            from model import CogVideoXTransformer3DModel
            from diffusers.utils import export_to_video, load_video
            
            #------Config------# 0 ~ 49, 50
            # text_timesteps = [0]
            text_timesteps = [i for i in range(50)]
            # cond_timesteps = [i for i in range(50, -1, -1)]
            cond_timesteps = [50]
            #------------------#
            
            model_id = "/home/nas4_user/kinamkim/checkpoint/cogvideox-5b"
            pipe = CogVideoXPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            pipe.transformer = CogVideoXTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=torch.bfloat16
            )
            pipe.to("cuda")
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            if args.lora_weight_path:
                pipe.load_lora_weights(args.lora_weight_path, adapter_name="cogvideox-lora")
                pipe.set_adapters(["cogvideox-lora"], [0.75])
            else:
                args.lora_weight_path = "output/base/dummy.safetensors"

        # Create validation_videos directory in the same folder as lora_weight_path
        lora_dir = args.lora_weight_path
        savedir = os.path.join(lora_dir, "validation_videos")
        if args.apply_target_noise_only:
            savedir = os.path.join(savedir, args.apply_target_noise_only)
        else:
            savedir = os.path.join(savedir, "None")
        dataset_name = "/".join(args.dataset_dir.split("/")[-2:])
        savedir = f"laboratory/outputs/{args.apply_target_noise_only}"
        os.makedirs(savedir, exist_ok=True)
        
        video_dir = os.path.join(args.dataset_dir, "videos")
        prompt_path = os.path.join(args.dataset_dir, "prompt.txt")
        with open(prompt_path, "r") as f:
            prompts = f.readlines()
        
        generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
        data_basename = os.path.join(*args.dataset_dir.split(os.sep)[-2:])
        for i, prompt in enumerate(prompts[:args.num_videos]):
            for text_timestep in text_timesteps:
                for cond_timestep in cond_timesteps:
                    print(f"Generating video {i+1}: {prompt[:100]}...")
                    print(f"text_t: {text_timestep} cond_t: {cond_timestep}")
                    video_path = os.path.join(video_dir, f"{i+1}.mp4")
                    
                    # Create directory with text_t and cond_t
                    current_savedir = os.path.join(savedir, f"text_{text_timestep}_cond_{cond_timestep}/{data_basename}/")
                    os.makedirs(current_savedir, exist_ok=True)
                    
                    # Use original video name for output
                    output_filename = f"{i+1}.mp4"
                    if os.path.exists(os.path.join(current_savedir, output_filename)):
                        print(f"Skipping path: {video_path}")
                        continue
                    print(f"video_path: {video_path}")
                    init_latents = process_video(pipe, 
                                                video_path, 
                                                torch.bfloat16, 
                                                generator, 
                                                args.height, 
                                                args.width,
                                                args.apply_target_noise_only)
                    
                    video = pipe(prompt, 
                                generator=generator, 
                                width=args.width, 
                                height=args.height, 
                                latents=init_latents,
                                apply_target_noise_only=args.apply_target_noise_only,
                                text_timestep_idx=text_timestep,
                                cond_timestep_idx=cond_timestep).frames[0]
                    export_to_video(video, os.path.join(current_savedir, output_filename))