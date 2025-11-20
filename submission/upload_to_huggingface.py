#!/usr/bin/env python3
"""
upload_to_huggingface.py - Upload checkpoint to HuggingFace Hub

This script uploads your Phase 2a checkpoint to HuggingFace for:
1. BTC reproducibility check
2. Sharing with teammates
3. Backup

Usage:
    python upload_to_huggingface.py --checkpoint_path ./checkpoint-clean \
                                     --repo_id thangquoc/zaic2025-phase2 \
                                     --token YOUR_HF_TOKEN
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, snapshot_download

def upload_checkpoint(checkpoint_path, repo_id, token, private=True):
    """Upload checkpoint to HuggingFace Hub"""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    print("="*60)
    print("Upload Checkpoint to HuggingFace Hub")
    print("="*60)
    print()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print()

    # Initialize API
    api = HfApi(token=token)

    # Create repository (if not exists)
    print("Step 1: Creating repository...")
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Failed to create repository: {e}")
        return False

    print()

    # Upload files
    print("Step 2: Uploading files...")
    try:
        # Upload entire folder
        api.upload_folder(
            folder_path=str(checkpoint_path),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload Phase 2a checkpoint for ZAIC 2025"
        )
        print("✓ All files uploaded successfully")
    except Exception as e:
        print(f"✗ Failed to upload: {e}")
        return False

    print()
    print("="*60)
    print("✅ Upload completed!")
    print("="*60)
    print()
    print(f"Repository URL: https://huggingface.co/{repo_id}")
    print()
    print("Next steps:")
    print(f"  1. Verify files at: https://huggingface.co/{repo_id}/tree/main")
    print(f"  2. Update training_code/README.md with this URL")
    print(f"  3. Test download with:")
    print(f"     python -c \"from huggingface_hub import snapshot_download; \\")
    print(f"                 snapshot_download(repo_id='{repo_id}', local_dir='./test-download')\"")
    print()

    return True


def download_checkpoint(repo_id, local_dir, token=None):
    """Download checkpoint from HuggingFace Hub"""

    print("="*60)
    print("Download Checkpoint from HuggingFace Hub")
    print("="*60)
    print()
    print(f"Repository: {repo_id}")
    print(f"Local directory: {local_dir}")
    print()

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token
        )
        print()
        print("✅ Download completed!")
        print(f"Checkpoint saved to: {local_dir}")
        return True
    except Exception as e:
        print(f"✗ Failed to download: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload/Download checkpoint to/from HuggingFace Hub")
    parser.add_argument("--mode", choices=["upload", "download"], required=True,
                      help="Upload or download checkpoint")
    parser.add_argument("--checkpoint_path", type=str,
                      help="Path to local checkpoint (for upload)")
    parser.add_argument("--repo_id", type=str, required=True,
                      help="HuggingFace repository ID (e.g., username/repo-name)")
    parser.add_argument("--local_dir", type=str,
                      help="Local directory to save checkpoint (for download)")
    parser.add_argument("--token", type=str,
                      help="HuggingFace token (or set HF_TOKEN env variable)")
    parser.add_argument("--private", action="store_true", default=True,
                      help="Make repository private (default: True)")

    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token and args.mode == "upload":
        print("Error: HuggingFace token required for upload")
        print("Get your token at: https://huggingface.co/settings/tokens")
        print()
        print("Usage:")
        print("  Option 1: Pass token as argument")
        print("    python upload_to_huggingface.py --mode upload --token YOUR_TOKEN ...")
        print()
        print("  Option 2: Set environment variable")
        print("    export HF_TOKEN=YOUR_TOKEN")
        print("    python upload_to_huggingface.py --mode upload ...")
        return

    if args.mode == "upload":
        if not args.checkpoint_path:
            print("Error: --checkpoint_path required for upload")
            return

        success = upload_checkpoint(
            checkpoint_path=args.checkpoint_path,
            repo_id=args.repo_id,
            token=token,
            private=args.private
        )

        if not success:
            exit(1)

    elif args.mode == "download":
        if not args.local_dir:
            print("Error: --local_dir required for download")
            return

        success = download_checkpoint(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token
        )

        if not success:
            exit(1)


if __name__ == "__main__":
    main()
