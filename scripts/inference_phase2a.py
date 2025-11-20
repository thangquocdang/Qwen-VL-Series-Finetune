#!/usr/bin/env python3
"""
Inference script for Phase 2a checkpoint (LoRA + Merger + Top-6 LLM layers)

This script properly loads:
1. Base Qwen2-VL-2B-Instruct model
2. LoRA adapter weights (all layers)
3. Merger weights (trained)
4. Top 6 LLM layer weights (trained, non-LoRA parts)

Usage:
    python scripts/inference_phase2a.py \
        --checkpoint /path/to/phase2a/checkpoint-latest \
        --image_path /path/to/image.jpg \
        --prompt "What do you see?"
"""

import argparse
import torch
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
import json


def set_seed(seed=42):
    """Set seeds for reproducibility (as per competition requirements)"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Set all random seeds to {seed}")


def load_phase2a_model(checkpoint_path, base_model_id="Qwen/Qwen2-VL-2B-Instruct", device="cuda"):
    """
    Load Phase 2a model with LoRA + Merger + Top-6 LLM layers

    Args:
        checkpoint_path: Path to Phase 2a checkpoint directory
        base_model_id: Base model to load
        device: Device to load model on

    Returns:
        model: Loaded model with all weights
        processor: Processor for the model
    """
    checkpoint_path = Path(checkpoint_path)
    print(f"\n{'='*60}")
    print(f"Loading Phase 2a Model from: {checkpoint_path}")
    print(f"{'='*60}\n")

    # 1. Load base model
    print("Step 1/4: Loading base model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    print(f"✓ Loaded base model: {base_model_id}")

    # 2. Load LoRA adapter
    print("\nStep 2/4: Loading LoRA adapter...")
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists():
        # Method 1: Use PeftModel (recommended)
        try:
            model = PeftModel.from_pretrained(model, str(checkpoint_path))
            print(f"✓ Loaded LoRA adapter using PeftModel.from_pretrained()")
        except Exception as e:
            print(f"Warning: PeftModel.from_pretrained() failed: {e}")
            print("Trying alternative method...")

            # Method 2: Manual loading
            from peft import LoraConfig, get_peft_model
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)

            lora_config = LoraConfig(**adapter_config)
            model = get_peft_model(model, lora_config)

            adapter_path = checkpoint_path / "adapter_model.safetensors"
            if adapter_path.exists():
                adapter_weights = load_file(str(adapter_path))
                set_peft_model_state_dict(model, adapter_weights)
                print(f"✓ Loaded LoRA adapter weights manually ({len(adapter_weights)} params)")
    else:
        print("⚠️  WARNING: No LoRA adapter found in checkpoint!")

    # 3. Load merger weights
    print("\nStep 3/4: Loading merger weights...")
    merger_weights_path = checkpoint_path / "merger_weights.bin"
    if merger_weights_path.exists():
        merger_weights = torch.load(merger_weights_path, map_location=device)

        missing_keys = []
        loaded_keys = []
        for name, param in model.named_parameters():
            if name in merger_weights:
                param.data.copy_(merger_weights[name])
                loaded_keys.append(name)
            elif "merger" in name:
                missing_keys.append(name)

        print(f"✓ Loaded {len(loaded_keys)} merger parameters")
        if missing_keys:
            print(f"⚠️  Missing {len(missing_keys)} merger parameters: {missing_keys[:3]}...")
    else:
        print("⚠️  WARNING: No merger weights found!")

    # 4. Load top-6 LLM layer weights (non-LoRA parts)
    print("\nStep 4/4: Loading top-6 LLM layer weights...")
    # Check for non_lora_state_dict.bin (if saved with save_non_lora_weights=True)
    non_lora_path = checkpoint_path / "non_lora_state_dict.bin"
    if non_lora_path.exists():
        non_lora_weights = torch.load(non_lora_path, map_location=device)

        # Load non-LoRA weights (LayerNorm, unfrozen layer weights, etc.)
        missing, unexpected = model.load_state_dict(non_lora_weights, strict=False)
        print(f"✓ Loaded non-LoRA weights")
        if missing:
            print(f"  Missing keys: {len(missing)} (expected for LoRA params)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("ℹ️  No separate non_lora_state_dict.bin found")
        print("   Top-6 layer weights may be included in LoRA adapter")
        print("   (This is normal if --save_non_lora_weights was not set)")

    # 5. Load processor
    print("\nStep 5/5: Loading processor...")
    processor = AutoProcessor.from_pretrained(
        str(checkpoint_path) if (checkpoint_path / "preprocessor_config.json").exists() else base_model_id,
        trust_remote_code=True
    )
    print("✓ Loaded processor")

    # Set model to eval mode
    model.eval()

    print(f"\n{'='*60}")
    print("✅ Model loaded successfully!")
    print(f"{'='*60}\n")

    return model, processor


def inference(model, processor, image_path, prompt, device="cuda"):
    """
    Run inference on an image with a prompt

    Args:
        model: Loaded model
        processor: Processor
        image_path: Path to image file
        prompt: Text prompt
        device: Device

    Returns:
        response: Model's response
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    print(f"\nRunning inference:")
    print(f"  Image: {image_path}")
    print(f"  Prompt: {prompt}")
    print()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Prepare messages (Qwen2-VL chat format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process vision info
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare inputs
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding for reproducibility
        )

    # Decode
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return response


def main():
    parser = argparse.ArgumentParser(description="Inference with Phase 2a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Phase 2a checkpoint directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                       help="Base model ID")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to image file")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for the image")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load model
    model, processor = load_phase2a_model(
        args.checkpoint,
        base_model_id=args.base_model,
        device=args.device
    )

    # Run inference
    response = inference(
        model,
        processor,
        args.image_path,
        args.prompt,
        device=args.device
    )

    # Print result
    print(f"\n{'='*60}")
    print("RESULT:")
    print(f"{'='*60}")
    print(f"\n{response}\n")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
