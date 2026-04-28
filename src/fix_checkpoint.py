import argparse
import os
import shutil
from huggingface_hub import snapshot_download

def fix_checkpoint(base_model: str, checkpoint_dir: str):
    print(f"Fixing checkpoint {checkpoint_dir} using base model {base_model}...")
    
    try:
        # Get the original files from HF cache
        base_path = snapshot_download(
            base_model, 
            allow_patterns=["*.py", "config.json", "preprocessor_config.json", "generation_config.json"]
        )
        
        # Copy them over to the output directory
        copied = []
        for filename in os.listdir(base_path):
            src_path = os.path.join(base_path, filename)
            if os.path.isfile(src_path):
                dst_path = os.path.join(checkpoint_dir, filename)
                shutil.copy(src_path, dst_path)
                copied.append(filename)
                
        print(f"Successfully copied: {', '.join(copied)}")
        print("Checkpoint is now ready for vLLM evaluation!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen3.5-0.8B-Base", help="Base model name")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint directory")
    args = parser.parse_args()
    
    fix_checkpoint(args.base, args.ckpt)
