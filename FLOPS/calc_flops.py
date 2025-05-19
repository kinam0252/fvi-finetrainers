import sys
import os
import re

def parse_profile_txt(profile_path):
    total_gflops = 0.0
    with open(profile_path, 'r') as f:
        lines = f.readlines()
    # GFLOPs 합계
    for line in lines:
        match = re.search(r'([\d\.]+)\s*$', line)
        if match and 'GFLOPs' not in line and '--' not in line:
            try:
                gflops = float(match.group(1))
                total_gflops += gflops
            except ValueError:
                continue
    total_flops = total_gflops * 1e9
    return total_gflops, total_flops

def save_flops_txt(profile_path, total_gflops, total_flops):
    out_path = os.path.join(os.path.dirname(profile_path), 'flops.txt')
    with open(out_path, 'w') as f:
        f.write(f"Total GFLOPs: {total_gflops:.3f}\n")
        f.write(f"Total FLOPs: {total_flops:.3e}\n")
    print(f"Saved: {out_path}")

def main():
    profile_path = "/home/nas4_user/kinamkim/video-in-context-lora/fvi-finetrainers/FLOPS/I2V/results/Controlnet/profile.txt"
    total_gflops, total_flops = parse_profile_txt(profile_path)
    save_flops_txt(profile_path, total_gflops, total_flops)

if __name__ == "__main__":
    main() 