#!/usr/bin/env python3
"""
Inference script for video question answering with multiple choices

Input JSON format:
{
    "data": [
        {
            "id": "testa_0001",
            "question": "Câu hỏi về video",
            "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
            "video_path": "path/to/video.mp4"
        }
    ]
}

Output JSON format:
{
    "data": [
        {
            "id": "testa_0001",
            "answer": "A"
        }
    ]
}

Usage:
    python scripts/inference_video_qa.py \
        --checkpoint /path/to/checkpoint-latest \
        --input_json test.json \
        --video_dir /path/to/videos \
        --output_json predictions.json
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re


def set_seed(seed=42):
    """Set seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path, base_model="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
    """Load Phase 2a model"""
    checkpoint_path = Path(checkpoint_path)
    print(f"\nLoading model from {checkpoint_path}...")

    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # Load LoRA adapter
    if (checkpoint_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        print("✓ Loaded LoRA adapter")

    # Load merger weights
    merger_path = checkpoint_path / "merger_weights.bin"
    if merger_path.exists():
        merger_weights = torch.load(merger_path, map_location=device)
        for name, param in model.named_parameters():
            if name in merger_weights:
                param.data.copy_(merger_weights[name])
        print(f"✓ Loaded {len(merger_weights)} merger parameters")

    # Load processor
    processor = AutoProcessor.from_pretrained(str(checkpoint_path), trust_remote_code=True)

    model.eval()
    print("✅ Model loaded!\n")

    return model, processor


def format_prompt(question, choices):
    """
    Format question and choices into prompt

    Example output:
        Câu hỏi: Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?

        A. Không có thông tin
        B. Dầu Giây Long Thành
        C. Đường Đỗ Xuân Hợp
        D. Xa Lộ Hà Nội

        Trả lời chỉ một chữ cái (A, B, C, hoặc D):
    """
    prompt = f"Câu hỏi: {question}\n\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "\nTrả lời chỉ một chữ cái (A, B, C, hoặc D):"
    return prompt


def extract_answer(response):
    """
    Extract answer letter (A/B/C/D) from model response

    Examples:
        "A" → "A"
        "The answer is B" → "B"
        "C. Đường Đỗ Xuân Hợp" → "C"
        "Đáp án là D" → "D"
    """
    # Try to find single letter A/B/C/D
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)

    # If not found, try to find at the beginning
    response_upper = response.strip().upper()
    if response_upper and response_upper[0] in 'ABCD':
        return response_upper[0]

    # Default fallback
    print(f"  ⚠️  Could not extract answer from: {response[:50]}...")
    return "A"  # Default to A if can't parse


def predict_video_qa(model, processor, video_path, question, choices, device="cuda"):
    """
    Predict answer for video question

    Args:
        model: Loaded model
        processor: Loaded processor
        video_path: Path to video file
        question: Question text
        choices: List of choice strings ["A. ...", "B. ...", ...]
        device: Device

    Returns:
        answer: Single letter "A", "B", "C", or "D"
        response: Full model response
    """
    # Format prompt
    prompt = format_prompt(question, choices)

    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Format input
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Short response for A/B/C/D
            do_sample=False  # Greedy for reproducibility
        )

    # Decode
    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    # Extract answer letter
    answer = extract_answer(response)

    return answer, response


def main():
    parser = argparse.ArgumentParser(description="Video QA inference for multiple choice questions")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Phase 2a checkpoint directory")
    parser.add_argument("--input_json", type=str, required=True,
                       help="Path to input JSON file")
    parser.add_argument("--video_dir", type=str, default="",
                       help="Base directory for videos (prepended to video_path)")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Path to output JSON file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Base model ID")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed responses")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load model
    model, processor = load_model(args.checkpoint, base_model=args.base_model, device=args.device)

    # Load input JSON
    print(f"Loading input from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    samples = input_data['data'] if 'data' in input_data else input_data
    print(f"Found {len(samples)} samples\n")

    # Video directory
    video_dir = Path(args.video_dir) if args.video_dir else Path("")

    # Process each sample
    results = []
    for sample in tqdm(samples, desc="Processing"):
        sample_id = sample['id']
        question = sample['question']
        choices = sample['choices']
        video_path = video_dir / sample['video_path']

        # Check video exists
        if not video_path.exists():
            print(f"\n⚠️  Video not found: {video_path}")
            # Default to A if video not found
            results.append({
                "id": sample_id,
                "answer": "A"
            })
            continue

        # Predict
        try:
            answer, response = predict_video_qa(
                model, processor, str(video_path), question, choices, device=args.device
            )

            results.append({
                "id": sample_id,
                "answer": answer
            })

            if args.verbose:
                print(f"\n{'='*60}")
                print(f"ID: {sample_id}")
                print(f"Question: {question[:80]}...")
                print(f"Response: {response}")
                print(f"Answer: {answer}")
                print(f"{'='*60}")

        except Exception as e:
            print(f"\n❌ Error processing {sample_id}: {e}")
            # Default to A on error
            results.append({
                "id": sample_id,
                "answer": "A"
            })

    # Save output
    output_data = {"data": results}
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(results)} predictions to {args.output_json}")

    # Print summary
    answer_counts = {}
    for result in results:
        ans = result['answer']
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    print("\nAnswer distribution:")
    for letter in ['A', 'B', 'C', 'D']:
        count = answer_counts.get(letter, 0)
        pct = 100 * count / len(results) if results else 0
        print(f"  {letter}: {count:4d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
