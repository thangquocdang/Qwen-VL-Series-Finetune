#!/usr/bin/env python3
"""
predict.py - Main inference script for Zalo AI Challenge 2025

Input: /data/test.json - Test cases in JSON format
Output: /result/submission.csv - Predictions in required format

Requirements:
- Model checkpoint at ./saved_models/checkpoint-latest/
- Test data at /data/test.json
- Output to /result/submission.csv
"""

import os
import sys
import json
import time
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re


# ========== Configuration ==========
CHECKPOINT_PATH = "./saved_models/checkpoint-latest"
INPUT_JSON = "/data/test.json"
OUTPUT_DIR = "/result"
OUTPUT_FILE = f"{OUTPUT_DIR}/submission.csv"
SEED = 42

# Optional: Download checkpoint from HuggingFace if not exists locally
# Set this to your HuggingFace repo ID (e.g., "thangquoc/zaic2025-phase2")
# Leave as None to use local checkpoint only (recommended for competition)
HUGGINGFACE_REPO_ID = os.environ.get("HF_CHECKPOINT_REPO", None)


# ========== Download Checkpoint from HuggingFace (Optional) ==========
def download_checkpoint_if_needed(checkpoint_path, repo_id=None):
    """
    Download checkpoint from HuggingFace if local checkpoint doesn't exist

    Args:
        checkpoint_path: Local path to checkpoint
        repo_id: HuggingFace repo ID (e.g., "username/repo-name")

    Returns:
        Path to checkpoint (either local or downloaded)
    """
    checkpoint_path = Path(checkpoint_path)

    # Check if local checkpoint exists
    if checkpoint_path.exists():
        print(f"✓ Using local checkpoint: {checkpoint_path}")
        return checkpoint_path

    # If no repo_id provided, fail
    if not repo_id:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} and no HuggingFace repo specified.\n"
            f"Either:\n"
            f"  1. Copy checkpoint to {checkpoint_path}, or\n"
            f"  2. Set HF_CHECKPOINT_REPO environment variable to download from HuggingFace"
        )

    # Download from HuggingFace
    print(f"⚠️  Local checkpoint not found at {checkpoint_path}")
    print(f"Downloading checkpoint from HuggingFace: {repo_id}")
    print("This may take a few minutes...")

    try:
        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(checkpoint_path),
            token=os.environ.get("HF_TOKEN")  # Optional: for private repos
        )

        print(f"✓ Checkpoint downloaded to: {checkpoint_path}")
        return Path(downloaded_path)

    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download checkpoints.\n"
            "Install with: pip install huggingface_hub"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download checkpoint from HuggingFace: {e}")


# ========== Set Seed for Reproducibility ==========
def seed_everything(seed=42):
    """Set all random seeds for reproducibility (competition requirement)"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Set all random seeds to {seed}")


# ========== Helper Functions ==========
def format_prompt(question, choices):
    """Format prompt to match training format"""
    prompt = f"<video>\n{question}\n\n"
    for choice in choices:
        prompt += f"{choice}\n"
    return prompt.rstrip()


def extract_answer(response):
    """Extract A/B/C/D from model response"""
    # Priority 1: "Đáp án: X" format (training format)
    match = re.search(r'(?:Đáp án|đáp án)[:\s]+([ABCD])', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Priority 2: Standalone A/B/C/D
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)

    # Priority 3: Starts with A/B/C/D
    if response.strip() and response.strip().upper()[0] in 'ABCD':
        return response.strip().upper()[0]

    # Default
    return "A"


def load_model(checkpoint_path, device="cuda"):
    """Load Phase 2a model (LoRA + Merger + Top-6 LLM layers)"""
    print(f"\n{'='*60}")
    print(f"Loading model from {checkpoint_path}...")
    print(f"{'='*60}\n")

    # Load base model
    print("Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    print("✓ Base model loaded")

    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    print("✓ LoRA adapter loaded")

    # Load merger weights
    print("Loading merger weights...")
    merger_path = Path(checkpoint_path) / "merger_weights.bin"
    if merger_path.exists():
        merger_weights = torch.load(merger_path, map_location=device)
        for name, param in model.named_parameters():
            if name in merger_weights:
                param.data.copy_(merger_weights[name])
        print(f"✓ Loaded {len(merger_weights)} merger parameters")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    print("✓ Processor loaded")

    model.eval()
    print(f"\n{'='*60}")
    print("✅ Model ready for inference!")
    print(f"{'='*60}\n")

    return model, processor


def predict_single(model, processor, video_path, question, choices, device="cuda"):
    """Predict answer for a single test case"""
    prompt = format_prompt(question, choices)

    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(video_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return extract_answer(response)


def main():
    """Main inference pipeline"""

    # Set seed
    seed_everything(SEED)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download checkpoint from HuggingFace if needed
    checkpoint_path = download_checkpoint_if_needed(CHECKPOINT_PATH, HUGGINGFACE_REPO_ID)

    # Timing: Load model
    print("\n[1/3] Loading model and resources...")
    t_load_start = time.time()
    model, processor = load_model(checkpoint_path)
    t_load_end = time.time()
    load_time = t_load_end - t_load_start
    print(f"⏱️  Model load time: {load_time:.2f} seconds ({load_time*1000:.0f} ms)\n")

    # Load test data
    print("[2/3] Loading test data...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_cases = test_data['data'] if 'data' in test_data else test_data
    print(f"✓ Loaded {len(test_cases)} test cases\n")

    # Inference
    print("[3/3] Running inference...")
    t_inference_start = time.time()

    results = []
    for sample in tqdm(test_cases, desc="Predicting"):
        sample_id = sample['id']
        question = sample['question']
        choices = sample['choices']
        video_path = Path("/data") / sample['video_path']

        # Skip if video not found
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}, defaulting to A")
            results.append({"id": sample_id, "answer": "A"})
            continue

        # Predict
        try:
            answer = predict_single(model, processor, str(video_path), question, choices)
            results.append({"id": sample_id, "answer": answer})
        except Exception as e:
            print(f"❌ Error processing {sample_id}: {e}")
            results.append({"id": sample_id, "answer": "A"})

    t_inference_end = time.time()
    inference_time = t_inference_end - t_inference_start
    total_time = t_load_end - t_load_start + inference_time

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Results saved to {OUTPUT_FILE}")
    print(f"{'='*60}\n")

    # Print timing summary
    print("TIMING SUMMARY:")
    print(f"  Model load time:     {load_time:.2f}s ({load_time*1000:.0f} ms)")
    print(f"  Inference time:      {inference_time:.2f}s ({inference_time*1000:.0f} ms)")
    print(f"  Total time:          {total_time:.2f}s ({total_time*1000:.0f} ms)")
    print(f"  Avg per sample:      {inference_time/len(test_cases):.2f}s ({inference_time*1000/len(test_cases):.0f} ms)")
    print()

    # Print answer distribution
    answer_counts = df['answer'].value_counts().sort_index()
    print("ANSWER DISTRIBUTION:")
    for ans, count in answer_counts.items():
        pct = 100 * count / len(results)
        print(f"  {ans}: {count:4d} ({pct:5.1f}%)")
    print()

    print("✅ Inference completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
