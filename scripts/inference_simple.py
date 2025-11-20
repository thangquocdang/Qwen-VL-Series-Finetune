#!/usr/bin/env python3
"""
Simple inference script for Phase 2a checkpoint

Quick usage:
    from scripts.inference_simple import load_model, predict

    # Load once
    model, processor = load_model("/path/to/checkpoint")

    # Predict multiple times
    result = predict(model, processor, "image.jpg", "What is this?")
    print(result)
"""

import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info


def load_model(checkpoint_path, base_model="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
    """
    Load Phase 2a model (LoRA + Merger + Top-6 layers)

    Args:
        checkpoint_path: Path to checkpoint folder
        base_model: Base model name
        device: 'cuda' or 'cpu'

    Returns:
        model, processor
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"Loading model from {checkpoint_path}...")

    # Load base model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # Load LoRA adapter (includes merged weights for Phase 2a)
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
    processor = AutoProcessor.from_pretrained(
        str(checkpoint_path),
        trust_remote_code=True
    )

    model.eval()
    print("✅ Model loaded!\n")

    return model, processor


def predict(model, processor, image_path, prompt, max_tokens=512, device="cuda"):
    """
    Predict on a single image

    Args:
        model: Loaded model
        processor: Loaded processor
        image_path: Path to image
        prompt: Question/prompt
        max_tokens: Max generation length
        device: Device

    Returns:
        str: Model's response
    """
    # Prepare message
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
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
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

    # Decode
    generated = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)
    ]
    response = processor.batch_decode(generated, skip_special_tokens=True)[0]

    return response


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("Usage: python inference_simple.py <checkpoint> <image> <prompt>")
        print('Example: python inference_simple.py ./checkpoint-latest image.jpg "What do you see?"')
        sys.exit(1)

    checkpoint, image, prompt = sys.argv[1:4]

    # Load model
    model, processor = load_model(checkpoint)

    # Predict
    result = predict(model, processor, image, prompt)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}\n")
