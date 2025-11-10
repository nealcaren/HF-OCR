---
viewer: false
tags: [uv-script, ocr, vision-language-model, document-processing]
---

# OCR UV Scripts

> Part of [uv-scripts](https://huggingface.co/uv-scripts) - ready-to-run ML tools powered by UV

Ready-to-run OCR scripts that work with `uv run` - no setup required!

## 🚀 Quick Start with HuggingFace Jobs

Run OCR on any dataset without needing your own GPU:

```bash
# Quick test with 10 samples
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 10
```

That's it! The script will:

- ✅ Process first 10 images from your dataset
- ✅ Add OCR results as a new `markdown` column
- ✅ Push the results to a new dataset
- 📊 View results at: `https://huggingface.co/datasets/[your-output-dataset]`

## 📋 Available Scripts

### LightOnOCR (`lighton-ocr.py`) ⚡ Good one to test first since it's small and fast!

Fast and compact OCR using [lightonai/LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025):

- ⚡ **Fastest**: 5.71 pages/sec on H100, ~6.25 images/sec on A100 with batch_size=4096
- 🎯 **Compact**: Only 1B parameters - quick to download and initialize
- 🌍 **Multilingual**: 3 vocabulary sizes for different use cases
- 📐 **LaTeX formulas**: Mathematical notation in LaTeX format
- 📊 **Table extraction**: Markdown table format
- 📝 **Document structure**: Preserves hierarchy and layout
- 🚀 **Production-ready**: 76.1% benchmark score, used in production

**Vocabulary sizes:**
- `151k`: Full vocabulary, all languages (default)
- `32k`: European languages, ~12% faster decoding
- `16k`: European languages, ~12% faster decoding

**Quick start:**
```bash
# Test on 100 samples with English text (32k vocab is fastest for European languages)
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \
    your-input-dataset your-output-dataset \
    --vocab-size 32k \
    --batch-size 32 \
    --max-samples 100

# Full production run on A100 (can handle huge batches!)
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/lighton-ocr.py \
    your-input-dataset your-output-dataset \
    --vocab-size 32k \
    --batch-size 4096 \
    --temperature 0.0
```

### DeepSeek-OCR (`deepseek-ocr-vllm.py`) ⭐ NEW

Advanced document OCR using [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) with visual-text compression:

- 📐 **LaTeX equations** - Mathematical formulas in LaTeX format
- 📊 **Tables** - Extracted as HTML/markdown
- 📝 **Document structure** - Headers, lists, formatting preserved
- 🖼️ **Image grounding** - Spatial layout with bounding boxes
- 🔍 **Complex layouts** - Multi-column and hierarchical structures
- 🌍 **Multilingual** - Multiple language support
- 🎚️ **Resolution modes** - 5 presets for speed/quality trade-offs
- 💬 **Prompt modes** - 5 presets for different OCR tasks
- ⚡ **Fast batch processing** - vLLM acceleration

**Resolution Modes:**
- `tiny` (512×512): Fast, 64 vision tokens
- `small` (640×640): Balanced, 100 vision tokens
- `base` (1024×1024): High quality, 256 vision tokens
- `large` (1280×1280): Maximum quality, 400 vision tokens
- `gundam` (dynamic): Adaptive multi-tile (default)

**Prompt Modes:**
- `document`: Convert to markdown with grounding (default)
- `image`: OCR any image with grounding
- `free`: Fast OCR without layout
- `figure`: Parse figures from documents
- `describe`: Detailed image descriptions

### RolmOCR (`rolm-ocr.py`)

Fast general-purpose OCR using [reducto/RolmOCR](https://huggingface.co/reducto/RolmOCR) based on Qwen2.5-VL-7B:

- 🚀 **Fast extraction** - Optimized for speed and efficiency
- 📄 **Plain text output** - Clean, natural text representation
- 💪 **General-purpose** - Works well on various document types
- 🔥 **Large context** - Handles up to 16K tokens
- ⚡ **Batch optimized** - Efficient processing with vLLM

### Nanonets OCR (`nanonets-ocr.py`)

State-of-the-art document OCR using [nanonets/Nanonets-OCR-s](https://huggingface.co/nanonets/Nanonets-OCR-s) that handles:

- 📐 **LaTeX equations** - Mathematical formulas preserved
- 📊 **Tables** - Extracted as HTML format
- 📝 **Document structure** - Headers, lists, formatting maintained
- 🖼️ **Images** - Captions and descriptions included
- ☑️ **Forms** - Checkboxes rendered as ☐/☑

### Nanonets OCR2 (`nanonets-ocr2.py`)

Next-generation Nanonets OCR using [nanonets/Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) with improved accuracy:

- 🎯 **Enhanced quality** - 3.75B parameters for superior OCR accuracy
- 📐 **LaTeX equations** - Mathematical formulas preserved in LaTeX format
- 📊 **Advanced tables** - Improved HTML table extraction
- 📝 **Document structure** - Headers, lists, formatting maintained
- 🖼️ **Smart image captions** - Intelligent descriptions and captions
- ☑️ **Forms** - Checkboxes rendered as ☐/☑
- 🌍 **Multilingual** - Enhanced language support
- 🔧 **Based on Qwen2.5-VL** - Built on state-of-the-art vision-language model

### SmolDocling (`smoldocling-ocr.py`)

Ultra-compact document understanding using [ds4sd/SmolDocling-256M-preview](https://huggingface.co/ds4sd/SmolDocling-256M-preview) with only 256M parameters:

- 🏷️ **DocTags format** - Efficient XML-like representation
- 💻 **Code blocks** - Preserves indentation and syntax
- 🔢 **Formulas** - Mathematical expressions with layout
- 📊 **Tables & charts** - Structured data extraction
- 📐 **Layout preservation** - Bounding boxes and spatial info
- ⚡ **Ultra-fast** - Tiny model size for quick inference

### NuMarkdown (`numarkdown-ocr.py`)

Advanced reasoning-based OCR using [numind/NuMarkdown-8B-Thinking](https://huggingface.co/numind/NuMarkdown-8B-Thinking) that analyzes documents before converting to markdown:

- 🧠 **Reasoning Process** - Thinks through document layout before generation
- 📊 **Complex Tables** - Superior table extraction and formatting
- 📐 **Mathematical Formulas** - Accurate LaTeX/math notation preservation
- 🔍 **Multi-column Layouts** - Handles complex document structures
- ✨ **Thinking Traces** - Optional inclusion of reasoning process with `--include-thinking`

### DoTS.ocr (`dots-ocr.py`)

Compact multilingual OCR using [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) with only 1.7B parameters:

- 🌍 **100+ Languages** - Extensive multilingual support
- 📝 **Simple OCR** - Clean text extraction (default mode)
- 📊 **Layout Analysis** - Optional structured output with bboxes and categories
- 📐 **Formula recognition** - LaTeX format support
- 🎯 **Compact** - Only 1.7B parameters, efficient on smaller GPUs
- 🔀 **Flexible prompts** - Switch between OCR, layout-all, and layout-only modes

### olmOCR2 (`olmocr2-vllm.py`)

High-quality document OCR using [allenai/olmOCR-2-7B-1025-FP8](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8) optimized with GRPO reinforcement learning:

- 🎯 **High accuracy** - 82.4 ± 1.1 on olmOCR-Bench (84.9% on math)
- 📐 **LaTeX equations** - Mathematical formulas in LaTeX format
- 📊 **Table extraction** - Structured table recognition
- 📑 **Multi-column layouts** - Complex document structures
- 🗜️ **FP8 quantized** - Efficient 8B model for faster inference
- 📜 **Degraded scans** - Works well on old/historical documents
- 📝 **Long text extraction** - Headers, footers, and full document content
- 🧩 **YAML metadata** - Structured front matter (language, rotation, content type)
- 🚀 **Based on Qwen2.5-VL-7B** - Fine-tuned with reinforcement learning


## 🆕 New Features

### Multi-Model Comparison Support

All scripts now include `inference_info` tracking for comparing multiple OCR models:

```bash
# First model
uv run rolm-ocr.py my-dataset my-dataset --max-samples 100

# Second model (appends to same dataset)
uv run nanonets-ocr.py my-dataset my-dataset --max-samples 100

# View all models used
python -c "import json; from datasets import load_dataset; ds = load_dataset('my-dataset'); print(json.loads(ds[0]['inference_info']))"
```

### Random Sampling

Get representative samples with the new `--shuffle` flag:

```bash
# Random 50 samples instead of first 50
uv run rolm-ocr.py ordered-dataset output --max-samples 50 --shuffle

# Reproducible random sampling
uv run nanonets-ocr.py dataset output --max-samples 100 --shuffle --seed 42
```

### Automatic Dataset Cards

Every OCR run now generates comprehensive dataset documentation including:
- Model configuration and parameters
- Processing statistics
- Column descriptions
- Reproduction instructions

## 💻 Usage Examples

### Run on HuggingFace Jobs (Recommended)

No GPU? No problem! Run on HF infrastructure:

```bash
# DeepSeek-OCR - Real-world example (National Library of Scotland handbooks)
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    NationalLibraryOfScotland/Britain-and-UK-Handbooks-Dataset \
    davanstrien/handbooks-deep-ocr \
    --max-samples 100 \
    --shuffle \
    --resolution-mode large

# DeepSeek-OCR - Fast testing with tiny mode
hf jobs uv run --flavor l4x1 \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    your-input-dataset your-output-dataset \
    --max-samples 10 \
    --resolution-mode tiny

# DeepSeek-OCR - Parse figures from scientific papers
hf jobs uv run --flavor a100-large \
    -s HF_TOKEN \
    -e UV_TORCH_BACKEND=auto \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/deepseek-ocr-vllm.py \
    scientific-papers figures-extracted \
    --prompt-mode figure

# Basic OCR job with Nanonets
hf jobs uv run --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    your-input-dataset your-output-dataset

# DoTS.ocr - Multilingual OCR with compact 1.7B model
hf jobs uv run --flavor a100-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/dots-ocr.py \
    davanstrien/ufo-ColPali \
    your-username/ufo-ocr \
    --batch-size 256 \
    --max-samples 1000 \
    --shuffle

# Real example with UFO dataset 🛸
hf jobs uv run \
    --flavor a10g-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    davanstrien/ufo-ColPali \
    your-username/ufo-ocr \
    --image-column image \
    --max-model-len 16384 \
    --batch-size 128

# Nanonets OCR2 - Next-gen quality with 3B model
hf jobs uv run \
    --flavor l4x1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr2.py \
    your-input-dataset \
    your-output-dataset \
    --batch-size 16

# NuMarkdown with reasoning traces for complex documents
hf jobs uv run \
    --flavor l4x4 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/numarkdown-ocr.py \
    your-input-dataset your-output-dataset \
    --max-samples 50 \
    --include-thinking \
    --shuffle

# olmOCR2 - High-quality OCR with YAML metadata
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/olmocr2-vllm.py \
    your-input-dataset your-output-dataset \
    --batch-size 16 \
    --max-samples 100

# Private dataset with custom settings
hf jobs uv run --flavor l40sx1 \
    --secrets HF_TOKEN \
    https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    private-input private-output \
    --private \
    --batch-size 32
```

### Python API

```python
from huggingface_hub import run_uv_job

job = run_uv_job(
    "https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py",
    args=["input-dataset", "output-dataset", "--batch-size", "16"],
    flavor="l4x1"
)
```

### Run Locally (Requires GPU)

```bash
# Clone and run
git clone https://huggingface.co/datasets/uv-scripts/ocr
cd ocr
uv run nanonets-ocr.py input-dataset output-dataset

# Or run directly from URL
uv run https://huggingface.co/datasets/uv-scripts/ocr/raw/main/nanonets-ocr.py \
    input-dataset output-dataset

# RolmOCR for fast text extraction
uv run rolm-ocr.py documents extracted-text
uv run rolm-ocr.py images texts --shuffle --max-samples 100  # Random sample

# Nanonets OCR2 for highest quality
uv run nanonets-ocr2.py documents ocr-results

```

## 📁 Works With

Any HuggingFace dataset containing images - documents, forms, receipts, books, handwriting.

## 🎛️ Configuration Options

### Common Options (All Scripts)

| Option                     | Default | Description                   |
| -------------------------- | ------- | ----------------------------- |
| `--image-column`           | `image` | Column containing images      |
| `--batch-size`             | `32`/`16`* | Images processed together     |
| `--max-model-len`          | `8192`/`16384`** | Max context length     |
| `--max-tokens`             | `4096`/`8192`** | Max output tokens      |
| `--gpu-memory-utilization` | `0.8`   | GPU memory usage (0.0-1.0)    |
| `--split`                  | `train` | Dataset split to process      |
| `--max-samples`            | None    | Limit samples (for testing)   |
| `--private`                | False   | Make output dataset private   |
| `--shuffle`                | False   | Shuffle dataset before processing |
| `--seed`                   | `42`    | Random seed for shuffling     |

*RolmOCR and DoTS use batch size 16
**RolmOCR uses 16384/8192

### Script-Specific Options

**DeepSeek-OCR**:
- `--resolution-mode`: Quality level - `tiny`, `small`, `base`, `large`, or `gundam` (default)
- `--prompt-mode`: Task type - `document` (default), `image`, `free`, `figure`, or `describe`
- `--prompt`: Custom OCR prompt (overrides prompt-mode)
- `--base-size`, `--image-size`, `--crop-mode`: Override resolution mode manually
- ⚠️ **Important for HF Jobs**: Add `-e UV_TORCH_BACKEND=auto` for proper PyTorch installation

**RolmOCR**:
- Output column is auto-generated from model name (e.g., `rolmocr_text`)
- Use `--output-column` to override the default name

**DoTS.ocr**:
- `--prompt-mode`: Choose `ocr` (default), `layout-all`, or `layout-only`
- `--custom-prompt`: Override with custom prompt text
- `--output-column`: Output column name (default: `markdown`)

💡 **Performance tip**: Increase batch size for faster processing (e.g., `--batch-size 256` on A100)
