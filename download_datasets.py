import os
import argparse
from huggingface_hub import snapshot_download
from pathlib import Path
from tqdm import tqdm

def download_dataset(dataset_name, save_dir):
    """
    Download a dataset from Hugging Face using snapshot_download and save it to the specified directory.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face
        save_dir (Path): Directory to save the dataset
    """
    try:
        # Create save directory if it doesn't exist
        save_path = save_dir / dataset_name.split('/')[-1]
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Download the dataset with progress bar
        with tqdm(total=1, desc=f"Downloading {dataset_name}", unit="dataset") as pbar:
            snapshot_download(
                repo_id=dataset_name,
                repo_type="dataset",
                local_dir=str(save_path),
                local_dir_use_symlinks=False
            )
            pbar.update(1)
        print(f"Successfully downloaded {dataset_name} to {save_path}")
        
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Download datasets from Hugging Face')
    parser.add_argument('--save_dir', type=str, default='datasets',
                      help='Directory to save the datasets (default: datasets)')
    
    args = parser.parse_args()
    save_dir = Path(args.save_dir)
    
    # List of datasets to download
    datasets_to_download = [
        "finetrainers/cakeify-smol",
        "finetrainers/crush-smol",
        "finetrainers/squish-pika",
        "finetrainers/3dgs-dissolve"
    ]
    
    # Create the main save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Download each dataset with overall progress bar
    for dataset_name in tqdm(datasets_to_download, desc="Overall Progress", unit="dataset"):
        download_dataset(dataset_name, save_dir)

if __name__ == "__main__":
    main()
