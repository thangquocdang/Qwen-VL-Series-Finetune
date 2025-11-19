"""
Simple Video QA Inference Example for Notebooks

Quick usage in Kaggle/Colab notebook:
    %run notebooks/inference_video_qa_example.py
"""

import json
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re


# ========== Configuration ==========
CHECKPOINT = "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
INPUT_JSON = "/kaggle/input/test-data/test.json"
VIDEO_DIR = "/kaggle/input/test-videos"
OUTPUT_JSON = "/kaggle/working/predictions.json"
SEED = 42

# ========== Setup ==========
print("Setting seed...")
import random, numpy as np
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ========== Load Model ==========
print(f"\nLoading model from {CHECKPOINT}...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, CHECKPOINT)
print("✓ LoRA loaded")

merger_path = Path(CHECKPOINT) / "merger_weights.bin"
if merger_path.exists():
    merger_weights = torch.load(merger_path, map_location="cuda")
    for name, param in model.named_parameters():
        if name in merger_weights:
            param.data.copy_(merger_weights[name])
    print(f"✓ Merger loaded ({len(merger_weights)} params)")

processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)
model.eval()
print("✅ Model ready!\n")


# ========== Helper Functions ==========
def format_prompt(question, choices):
    """Format question and choices"""
    prompt = f"Câu hỏi: {question}\n\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "\nTrả lời chỉ một chữ cái (A, B, C, hoặc D):"
    return prompt


def extract_answer(response):
    """Extract A/B/C/D from response"""
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)
    response_upper = response.strip().upper()
    if response_upper and response_upper[0] in 'ABCD':
        return response_upper[0]
    return "A"  # Default


def predict(video_path, question, choices):
    """Predict answer for one sample"""
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
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return extract_answer(response), response


# ========== Load Data ==========
print(f"Loading {INPUT_JSON}...")
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)

samples = data['data'] if 'data' in data else data
print(f"Found {len(samples)} samples\n")

# ========== Process ==========
results = []
video_base = Path(VIDEO_DIR)

for sample in tqdm(samples, desc="Predicting"):
    video_path = video_base / sample['video_path']

    if not video_path.exists():
        print(f"⚠️  Not found: {video_path}")
        results.append({"id": sample['id'], "answer": "A"})
        continue

    try:
        answer, response = predict(video_path, sample['question'], sample['choices'])
        results.append({"id": sample['id'], "answer": answer})
    except Exception as e:
        print(f"❌ Error {sample['id']}: {e}")
        results.append({"id": sample['id'], "answer": "A"})

# ========== Save ==========
output = {"data": results}
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved to {OUTPUT_JSON}")

# ========== Summary ==========
answer_counts = {}
for r in results:
    ans = r['answer']
    answer_counts[ans] = answer_counts.get(ans, 0) + 1

print("\nAnswer distribution:")
for letter in ['A', 'B', 'C', 'D']:
    count = answer_counts.get(letter, 0)
    pct = 100 * count / len(results)
    print(f"  {letter}: {count:4d} ({pct:5.1f}%)")

print("\n🎉 Done!")
