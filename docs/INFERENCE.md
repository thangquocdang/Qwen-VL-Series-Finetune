# Inference Guide for Phase 2a Model

This guide shows how to load and use the Phase 2a checkpoint (with LoRA + Merger + Top-6 LLM layers) for inference.

## Understanding Phase 2a Checkpoint

Phase 2a checkpoint contains:
1. **LoRA adapter** (`adapter_model.safetensors`) - LoRA weights for all 28 LLM layers
2. **Merger weights** (`merger_weights.bin`) - Trained visual-to-text merger
3. **Top-6 LLM layers** - Trained weights for layers 22-27 (included in adapter or non_lora_state_dict)

## Method 1: Simple Script (Recommended)

### Quick Start

```bash
python scripts/inference_simple.py \
    /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest \
    /path/to/image.jpg \
    "Phân loại biển báo trong ảnh này"
```

### Python Usage

```python
from scripts.inference_simple import load_model, predict

# Load model once
model, processor = load_model(
    "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
)

# Predict multiple times
result = predict(model, processor, "image1.jpg", "Biển báo gì?")
print(result)

result = predict(model, processor, "image2.jpg", "Mô tả biển báo")
print(result)
```

## Method 2: Detailed Script (Full Control)

For more control over loading and generation:

```bash
python scripts/inference_phase2a.py \
    --checkpoint /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest \
    --image_path /path/to/image.jpg \
    --prompt "Phân loại biển báo trong ảnh này" \
    --seed 42 \
    --device cuda
```

## Method 3: Notebook / Interactive

```python
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info

# Set seed for reproducibility (competition requirement)
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

checkpoint = "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"

# 1. Load base model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# 2. Load LoRA adapter (automatically loads Phase 2a weights)
model = PeftModel.from_pretrained(model, checkpoint)

# 3. Load merger weights
merger_weights = torch.load(f"{checkpoint}/merger_weights.bin", map_location="cuda")
for name, param in model.named_parameters():
    if name in merger_weights:
        param.data.copy_(merger_weights[name])

# 4. Load processor
processor = AutoProcessor.from_pretrained(checkpoint)

model.eval()
print("✅ Model loaded!")

# 5. Inference function
def predict(image_path, prompt):
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
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
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]

# 6. Use it
result = predict("/path/to/image.jpg", "Phân loại biển báo trong ảnh này")
print(result)
```

## Batch Inference (Process Multiple Images)

```python
from scripts.inference_simple import load_model, predict
from pathlib import Path
import pandas as pd

# Load model once
model, processor = load_model("./checkpoint-latest")

# Process all images in a directory
image_dir = Path("/kaggle/input/test-images")
results = []

for image_path in image_dir.glob("*.jpg"):
    response = predict(
        model,
        processor,
        str(image_path),
        "Phân loại biển báo trong ảnh này"
    )
    results.append({
        "image": image_path.name,
        "prediction": response
    })

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)
print(f"Processed {len(results)} images")
```

## Competition Submission Script

```python
from scripts.inference_simple import load_model, predict
import pandas as pd
from pathlib import Path

# Set seed (competition requirement)
import torch, random, numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load model
checkpoint = "/kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest"
model, processor = load_model(checkpoint)

# Load test data
test_df = pd.read_csv("/kaggle/input/test.csv")
test_images = Path("/kaggle/input/test-images")

# Predict
predictions = []
for idx, row in test_df.iterrows():
    image_path = test_images / row['image_filename']

    # Get prediction
    response = predict(
        model,
        processor,
        str(image_path),
        "Phân loại biển báo giao thông trong ảnh này. Trả lời tên biển báo."
    )

    predictions.append({
        'id': row['id'],
        'prediction': response
    })

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(test_df)} images")

# Save submission
submission = pd.DataFrame(predictions)
submission.to_csv("submission.csv", index=False)
print("✅ Submission saved!")
```

## Advanced: Merge LoRA for Faster Inference

If you want to merge LoRA weights into base model (faster inference, but larger model):

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

# Load model with LoRA
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, checkpoint_path)

# Merge LoRA into base weights
model = model.merge_and_unload()

# Load merger weights
merger_weights = torch.load(f"{checkpoint}/merger_weights.bin")
for name, param in model.named_parameters():
    if name in merger_weights:
        param.data.copy_(merger_weights[name])

# Now model has no LoRA adapter, just merged weights
# Inference is faster but model is larger
model.save_pretrained("./merged_model")
```

## Important Notes

### Reproducibility
Always set seed before inference (competition requirement):
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### Memory Optimization

**FP16 (Recommended for GPU):**
```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,  # Half precision
    device_map="cuda"
)
```

**4-bit/8-bit Quantization (For limited GPU memory):**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="cuda"
)
```

### Generation Parameters

**For reproducibility (deterministic):**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=False,  # Greedy decoding
)
```

**For creative responses:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)
```

## Troubleshooting

### Error: "No module named 'qwen_vl_utils'"
```bash
pip install qwen-vl-utils
```

### Error: "CUDA out of memory"
- Use FP16: `torch_dtype=torch.float16`
- Use 4-bit quantization
- Reduce `max_new_tokens`
- Process images one at a time

### Error: "LoRA adapter not found"
Make sure checkpoint path is correct:
```bash
ls /kaggle/working/checkpoints/zac_qwen2vl_phase2a/checkpoint-latest/
# Should contain: adapter_config.json, adapter_model.safetensors, merger_weights.bin
```

### Model outputs random text
- Make sure you loaded merger weights correctly
- Check that Phase 2a training completed successfully
- Verify checkpoint is not corrupted

## Performance Tips

1. **Batch processing**: Load model once, process many images
2. **FP16**: Use half precision on GPU
3. **Compile**: Use `torch.compile(model)` for PyTorch 2.0+
4. **Flash Attention**: Enable for faster inference (if supported)

```python
# Enable Flash Attention 2 (requires flash-attn package)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="cuda",
    attn_implementation="flash_attention_2"  # Faster!
)
```

## Verify Model Loading

Print trainable params to verify:
```python
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / Total: {total:,}")
print(f"Trainable %: {100 * trainable / total:.2f}%")
```

For Phase 2a inference, all params should be **non-trainable** (model.eval() mode).
