import torch
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video generation validation script")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["ltxvideo", "cogvideo"],
        required=True,
        help="Type of model to use for validation"
    )
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        required=True,
        help="Path to LoRA weights"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to text file containing prompts"
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
        default=768,
        help="Width of the video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the video"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=30,
        help="Number of videos to generate"
    )
    return parser.parse_args()


args = parse_args()


if __name__ == "__main__":
    if args.model_type == "ltxvideo":
        from diffusers import LTXPipeline
        from diffusers.utils import export_to_video
        pipe = LTXPipeline.from_pretrained(
            "Lightricks/LTX-Video", torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe.load_lora_weights(args.lora_weight_path, adapter_name="ltxv-lora")
        pipe.set_adapters(["ltxv-lora"], [0.75])

    elif args.model_type == "cogvideo":
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16
        ).to("cuda")
        pipe.load_lora_weights(args.lora_weight_path, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [0.75])

    with open(args.prompt_file, "r") as f:
        prompts = f.readlines()

    savedir = args.lora_weight_path + "/validation_videos"
    os.makedirs(savedir, exist_ok=True)
    generator = torch.Generator("cuda").manual_seed(args.seed)
    for i, prompt in enumerate(prompts[:args.num_videos]):
        print(f"Generating video {i+1}: {prompt[:100]}...")
        video = pipe(prompt, generator=generator, width=args.width, height=args.height).frames[0]
        export_to_video(video, os.path.join(savedir, f"output_{i}.mp4"))