import argparse
from safetensors.torch import load_file
from collections import defaultdict

def count_lora_params(lora_weight_path):
    """Count actual LoRA parameters from the safetensors file."""
    # Load LoRA weights
    lora_state_dict = load_file(lora_weight_path)
    
    # Analyze the structure of the weights
    print("Analyzing LoRA weight structure...")
    
    # Group parameters by their base path (excluding the lora_A/B suffix)
    param_groups = defaultdict(list)
    for key in lora_state_dict.keys():
        if 'lora_A' in key or 'lora_B' in key:
            # Remove lora_A/B suffix to group related parameters
            base_key = key.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
            param_groups[base_key].append(key)
    
    # Count parameters
    total_params = 0
    print("\nParameter counts by module:")
    for base_key, related_keys in param_groups.items():
        module_params = 0
        for key in related_keys:
            params = lora_state_dict[key].numel()
            module_params += params
            print(f"{key}: {params:,} parameters")
        total_params += module_params
        print(f"Total for {base_key}: {module_params:,} parameters\n")
    
    return total_params

def main():
    parser = argparse.ArgumentParser(description="Count LoRA parameters from safetensors file")
    parser.add_argument(
        "--lora_weight_path",
        type=str,
        required=False,
        default="/home/nas4_user/kinamkim/video-in-context-lora/fvi-finetrainers/output/360-cogvideo-front-4-noise-none-only25/pytorch_lora_weights.safetensors",
        help="Path to LoRA weights (safetensors file)"
    )
    args = parser.parse_args()

    # Count and print LoRA parameters
    lora_params = count_lora_params(args.lora_weight_path)
    print(f"\nLoRA Configuration:")
    print(f"LoRA weight path: {args.lora_weight_path}")
    print(f"Total LoRA parameters: {lora_params:,}")
    print(f"Total LoRA parameters (MB): {lora_params * 2 / (1024 * 1024):.2f}")

if __name__ == "__main__":
    main() 