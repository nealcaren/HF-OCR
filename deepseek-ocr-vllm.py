#!/usr/bin/env python3
"""
DeepSeek-OCR processing script using vLLM for batch inference on Hugging Face Hub.
Fixed dependency issues with stable package versions.
"""

# /// script
# dependencies = [
#   "datasets",
#   "huggingface-hub",
#   "pillow",
#   "torch",
#   "vllm>=0.6.0",
#   "tqdm",
#   "toolz",
# ]
# ///

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, login
from PIL import Image
from tqdm import tqdm
from vllm import LLM, SamplingParams


def check_cuda_availability():
    """Check if CUDA is available and log GPU info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    gpu_count = torch.cuda.device_count()
    print(f"✓ CUDA available with {gpu_count} GPU(s)")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    return True


def image_to_base64_uri(image_path: str) -> str:
    """Convert an image to a base64 data URI."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode to base64
            base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""


def make_ocr_message(image: Image.Image, prompt: str = "Please perform OCR on this image and return the result in Markdown format.") -> str:
    """Create a message with image data URI for the model."""
    # Convert PIL Image to base64
    buffer = io.BytesIO()

    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    image.save(buffer, format="PNG")
    buffer.seek(0)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    image_uri = f"data:image/png;base64,{base64_str}"

    return f"<image>{image_uri}</image>\n{prompt}"


def create_dataset_card(dataset_name: str, source_dataset: str, model_name: str, resolution_mode: str) -> str:
    """Create a dataset card for the output dataset."""
    return f"""---
license: mit
tags:
  - ocr
  - deepseek
  - vllm
  - markdown
---

# {dataset_name}

OCR results from {source_dataset} processed with {model_name} using vLLM.

## Processing Details
- **Source Dataset**: {source_dataset}
- **Model**: {model_name}
- **Resolution Mode**: {resolution_mode}
- **Processing Framework**: vLLM (batch inference)

## Dataset Structure
- `image`: Original image from source dataset
- `ocr_markdown`: OCR output in Markdown format

## Usage
```python
from datasets import load_dataset
ds = load_dataset("YOUR_USERNAME/{dataset_name}")
```
"""


def main():
    parser = argparse.ArgumentParser(description="Process OCR benchmark with DeepSeek-OCR using vLLM")
    parser.add_argument("source_dataset", help="Source dataset on HF Hub (e.g., NealCaren/InkBench)")
    parser.add_argument("output_dataset", help="Output dataset name on HF Hub")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for vLLM processing")
    parser.add_argument("--resolution-mode", default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Resolution mode for OCR")
    parser.add_argument("--prompt-mode", default="free", help="Prompt mode for OCR")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens to generate")

    args = parser.parse_args()

    # Check CUDA availability
    check_cuda_availability()

    # Authenticate with HF Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    print(f"\n📦 Loading source dataset: {args.source_dataset}")
    try:
        dataset = load_dataset(args.source_dataset, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"✓ Loaded {len(dataset)} samples")

    # Initialize vLLM with DeepSeek-OCR model
    print("\n🚀 Initializing vLLM with deepseek-ai/deepseek-vl2...")

    try:
        llm = LLM(
            model="deepseek-ai/deepseek-vl2",
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=False,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("\nTrying alternative model initialization...")
        try:
            llm = LLM(
                model="deepseek-ai/deepseek-vl2",
                gpu_memory_utilization=0.8,
                trust_remote_code=True,
            )
            print("✓ Model loaded with fallback settings")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            sys.exit(1)

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    # Process dataset
    print(f"\n📊 Processing {len(dataset)} samples with batch size {args.batch_size}...")

    ocr_results = []

    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
        batch = dataset[i:i + args.batch_size]

        # Prepare messages for batch
        messages = []
        images = []

        for j, sample in enumerate(zip(*batch.values())):
            sample_dict = {k: v for k, v in zip(batch.keys(), sample)}

            # Handle different image column names
            image = None
            for col in ["image", "images", "img"]:
                if col in sample_dict:
                    img_data = sample_dict[col]
                    if isinstance(img_data, Image.Image):
                        image = img_data
                    elif isinstance(img_data, dict) and "bytes" in img_data:
                        image = Image.open(io.BytesIO(img_data["bytes"]))
                    else:
                        try:
                            image = Image.open(img_data)
                        except:
                            pass
                    break

            if image is None:
                print(f"Warning: Could not load image for sample {i + j}")
                ocr_results.append({"ocr_markdown": "[Error: Image could not be loaded]"})
                continue

            # Create OCR message
            prompt = "Please perform OCR on this image and return the result in Markdown format. Include all text, tables, equations, and structural information."
            message = make_ocr_message(image, prompt)
            messages.append({"role": "user", "content": message})
            images.append(image)

        # Run vLLM inference
        if messages:
            try:
                outputs = llm.generate([msg["content"] for msg in messages], sampling_params)
                for output in outputs:
                    ocr_results.append({
                        "ocr_markdown": output.outputs[0].text if output.outputs else ""
                    })
            except Exception as e:
                print(f"Error during inference: {e}")
                for _ in messages:
                    ocr_results.append({"ocr_markdown": f"[Error: {str(e)}]"})

    # Create output dataset
    print("\n💾 Creating output dataset...")

    output_dataset = dataset.add_column("ocr_markdown", [r["ocr_markdown"] for r in ocr_results])

    # Push to Hub
    print(f"\n🚀 Pushing dataset to {args.output_dataset}...")
    try:
        output_dataset.push_to_hub(args.output_dataset, private=False)
        print(f"✓ Dataset pushed to https://huggingface.co/datasets/{args.output_dataset}")

        # Create and push dataset card
        api = HfApi()
        dataset_card_content = create_dataset_card(
            args.output_dataset,
            args.source_dataset,
            "deepseek-ai/deepseek-vl2",
            args.resolution_mode
        )
        try:
            api.upload_file(
                path_or_fileobj=dataset_card_content.encode(),
                path_in_repo="README.md",
                repo_id=args.output_dataset,
                repo_type="dataset",
            )
            print("✓ Dataset card created")
        except Exception as e:
            print(f"Warning: Could not upload dataset card: {e}")
    except Exception as e:
        print(f"Error pushing to hub: {e}")
        sys.exit(1)

    print("\n✨ Processing complete!")


if __name__ == "__main__":
    main()
