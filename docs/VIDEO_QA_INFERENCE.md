# Video Question Answering Inference Guide

Guide for running inference on video multiple-choice QA datasets.

## Input Format

Your JSON file should follow this format:

```json
{
    "data": [
        {
            "id": "testa_0001",
            "question": "Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường nào?",
            "choices": [
                "A. Không có thông tin",
                "B. Dầu Giây Long Thành",
                "C. Đường Đỗ Xuân Hợp",
                "D. Xa Lộ Hà Nội"
            ],
            "video_path": "public_test/videos/efc9909e_908_clip_001_0000_0009_Y.mp4"
        },
        {
            "id": "testa_0002",
            "question": "Theo trong video, nếu ô tô đi hướng chếch sang phải là hướng vào đường Xa Lộ Hà Nội. Đúng hay sai?",
            "choices": [
                "A. Đúng",
                "B. Sai"
            ],
            "video_path": "public_test/videos/efc9909e_908_clip_001_0000_0009_Y.mp4"
        }
    ]
}
```

## Output Format

The output will be:

```json
{
    "data": [
        {
            "id": "testa_0001",
            "answer": "D"
        },
        {
            "id": "testa_0002",
            "answer": "A"
        }
    ]
}
```

## Method 1: Command Line Script (Recommended)

### Basic Usage

```bash
python scripts/inference_video_qa.py \
    --checkpoint /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest \
    --input_json /kaggle/input/test-data/test.json \
    --video_dir /kaggle/input/test-videos \
    --output_json /kaggle/working/predictions.json \
    --seed 42
```

### With Verbose Output

To see detailed responses for each question:

```bash
python scripts/inference_video_qa.py \
    --checkpoint ./checkpoint-latest \
    --input_json test.json \
    --video_dir ./videos \
    --output_json predictions.json \
    --verbose
```

### Arguments

- `--checkpoint`: Path to Phase 2a checkpoint directory (required)
- `--input_json`: Path to input JSON file (required)
- `--video_dir`: Base directory for videos (prepended to video_path in JSON)
- `--output_json`: Path to output JSON file (required)
- `--base_model`: Base model ID (default: "Qwen/Qwen2-VL-2B-Instruct")
- `--device`: Device to run on (default: "cuda")
- `--seed`: Random seed for reproducibility (default: 42)
- `--verbose`: Print detailed responses for each sample

## Method 2: Notebook/Interactive

### Copy-Paste Ready Code

```python
import json
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re

# ========== Config ==========
CHECKPOINT = "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
INPUT_JSON = "/kaggle/input/test-data/test.json"
VIDEO_DIR = "/kaggle/input/test-videos"
OUTPUT_JSON = "/kaggle/working/predictions.json"

# ========== Set Seed ==========
import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ========== Load Model ==========
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, CHECKPOINT)

# Load merger
merger_weights = torch.load(f"{CHECKPOINT}/merger_weights.bin", map_location="cuda")
for name, param in model.named_parameters():
    if name in merger_weights:
        param.data.copy_(merger_weights[name])

processor = AutoProcessor.from_pretrained(CHECKPOINT)
model.eval()
print("✅ Model loaded!")

# ========== Helper Functions ==========
def format_prompt(question, choices):
    prompt = f"Câu hỏi: {question}\n\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "\nTrả lời chỉ một chữ cái (A, B, C, hoặc D):"
    return prompt

def extract_answer(response):
    match = re.search(r'\b([ABCD])\b', response.upper())
    if match:
        return match.group(1)
    if response.strip().upper()[0] in 'ABCD':
        return response.strip().upper()[0]
    return "A"

def predict(video_path, question, choices):
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
with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    data = json.load(f)
samples = data['data'] if 'data' in data else data

# ========== Process ==========
results = []
video_base = Path(VIDEO_DIR)

for sample in tqdm(samples):
    video_path = video_base / sample['video_path']

    if not video_path.exists():
        results.append({"id": sample['id'], "answer": "A"})
        continue

    try:
        answer, _ = predict(video_path, sample['question'], sample['choices'])
        results.append({"id": sample['id'], "answer": answer})
    except Exception as e:
        print(f"Error {sample['id']}: {e}")
        results.append({"id": sample['id'], "answer": "A"})

# ========== Save ==========
output = {"data": results}
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(results)} predictions to {OUTPUT_JSON}")
```

### Or Use Pre-made Notebook Script

```python
# In Kaggle/Colab notebook, just run:
%run notebooks/inference_video_qa_example.py
```

Then edit the config variables in the script.

## Method 3: Programmatic API

```python
from scripts.inference_video_qa import load_model, predict_video_qa

# Load model once
model, processor = load_model(
    "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
)

# Predict for single sample
answer, response = predict_video_qa(
    model=model,
    processor=processor,
    video_path="/path/to/video.mp4",
    question="Theo trong video, biển báo này là gì?",
    choices=[
        "A. Cấm đi ngược chiều",
        "B. Dừng lại",
        "C. Rẽ phải",
        "D. Không rõ"
    ]
)

print(f"Answer: {answer}")
print(f"Response: {response}")
```

## Features

### Answer Extraction

The script automatically extracts A/B/C/D from various response formats:

- `"A"` → A
- `"The answer is B"` → B
- `"C. Đường Đỗ Xuân Hợp"` → C
- `"Đáp án là D vì..."` → D
- `"Tôi nghĩ câu trả lời là A"` → A

### Error Handling

- If video file not found → defaults to "A"
- If model crashes → defaults to "A"
- If answer can't be parsed → defaults to "A"

### Progress Tracking

- Uses `tqdm` progress bar
- Shows processing speed (samples/sec)
- ETA for completion

## Kaggle Competition Example

```python
# Full pipeline for Kaggle competition
import json
import torch
from pathlib import Path
from scripts.inference_video_qa import load_model, predict_video_qa
from tqdm import tqdm

# Set seed (competition requirement)
import random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
CHECKPOINT = "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
TEST_JSON = "/kaggle/input/zac2025-test/test.json"
VIDEO_DIR = "/kaggle/input/zac2025-test/videos"
SUBMISSION = "/kaggle/working/submission.json"

# Load model
print("Loading model...")
model, processor = load_model(CHECKPOINT)

# Load test data
with open(TEST_JSON) as f:
    test_data = json.load(f)

# Process
results = []
for sample in tqdm(test_data['data'], desc="Processing"):
    video_path = Path(VIDEO_DIR) / sample['video_path']

    answer, _ = predict_video_qa(
        model, processor,
        str(video_path),
        sample['question'],
        sample['choices']
    )

    results.append({
        "id": sample['id'],
        "answer": answer
    })

# Save submission
with open(SUBMISSION, 'w', encoding='utf-8') as f:
    json.dump({"data": results}, f, ensure_ascii=False, indent=2)

print(f"✅ Submission saved to {SUBMISSION}")
```

## Performance Tips

### 1. Batch Processing (Faster)

If you have limited time, you can process videos in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def process_sample(sample):
    video_path = Path(VIDEO_DIR) / sample['video_path']
    answer, _ = predict_video_qa(model, processor, str(video_path),
                                  sample['question'], sample['choices'])
    return {"id": sample['id'], "answer": answer}

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_sample, samples))
```

### 2. FP16 (Already enabled by default)

The model loads in FP16 by default for faster inference.

### 3. Reduce max_new_tokens

For multiple choice, we only need A/B/C/D:

```python
# In predict_video_qa(), change:
max_new_tokens=128  # Default
# to:
max_new_tokens=32  # Faster, sufficient for A/B/C/D
```

### 4. Compile Model (PyTorch 2.0+)

```python
model = torch.compile(model)  # Faster after warmup
```

## Troubleshooting

### "Video file not found"

Make sure `video_dir` is set correctly. The script prepends `video_dir` to `video_path`:

```
Full path = video_dir / video_path
          = /kaggle/input/videos / public_test/videos/xxx.mp4
```

### "CUDA out of memory"

Reduce batch size or use CPU:

```bash
python scripts/inference_video_qa.py ... --device cpu
```

### "Wrong video format"

Make sure videos are in supported format (.mp4, .avi, etc.). The model uses `decord` to read videos.

### "Model gives random answers"

- Check that Phase 2a training completed successfully
- Verify checkpoint has all files: `adapter_model.safetensors`, `merger_weights.bin`
- Try verbose mode to see raw responses

## Verify Results

After inference, check the answer distribution:

```python
import json
with open('predictions.json') as f:
    data = json.load(f)

from collections import Counter
answers = [d['answer'] for d in data['data']]
print(Counter(answers))
# Example output: Counter({'A': 120, 'B': 95, 'C': 88, 'D': 102})
```

If one answer dominates (e.g., 95% are "A"), something may be wrong.

## Advanced: Custom Prompts

You can customize the prompt format in `format_prompt()`:

```python
def format_prompt(question, choices):
    # Vietnamese style
    prompt = f"Dựa vào video, hãy trả lời câu hỏi sau:\n\n{question}\n\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "\nChọn đáp án đúng (A/B/C/D):"
    return prompt
```

Or add few-shot examples:

```python
def format_prompt(question, choices):
    prompt = """Ví dụ:
Câu hỏi: Biển báo trong video là gì?
A. Cấm rẽ trái
B. Cấm rẽ phải
C. Cho phép rẽ phải
D. Không rõ
Trả lời: B

Bây giờ hãy trả lời:
"""
    prompt += f"Câu hỏi: {question}\n"
    for choice in choices:
        prompt += f"{choice}\n"
    prompt += "Trả lời:"
    return prompt
```

## Competition Checklist

Before submitting:

- [ ] Set seed to 42 (reproducibility requirement)
- [ ] Use Phase 2a checkpoint (best performance)
- [ ] Verify all videos can be read
- [ ] Check output format matches submission template
- [ ] Verify no errors in processing
- [ ] Check answer distribution (not all same letter)
- [ ] Test on sample before full test set
- [ ] Save submission.json in correct format

## Example Output

```
Loading model from /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest...
✓ Loaded LoRA adapter
✓ Loaded 6 merger parameters
✅ Model loaded!

Loading input from /kaggle/input/test-data/test.json...
Found 405 samples

Processing: 100%|████████████| 405/405 [12:35<00:00,  1.87s/it]

✅ Saved 405 predictions to /kaggle/working/predictions.json

Answer distribution:
  A:  112 (27.7%)
  B:   98 (24.2%)
  C:   95 (23.5%)
  D:  100 (24.7%)
```
