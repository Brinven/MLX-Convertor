# MLX Model Converter

A GUI application for converting HuggingFace models to Apple MLX format with built-in testing capabilities.

> **Apple Silicon Required** — This app requires a Mac with Apple Silicon (M1/M2/M3/M4). MLX does not support Windows or Linux.

## Installation

```bash
git clone <your-repo-url>
cd mlx-converter
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

```bash
python3 app.py
```

This opens a web UI in your browser with the following tabs:

### Convert

1. Enter a HuggingFace model path (e.g., `LiquidAI/LFM2-1.2B-RAG`)
2. Optionally set an output name
3. Choose quantization: **4-bit** (smallest), **8-bit** (balanced), or **bf16** (full precision)
4. Click **Convert Model**

Converted models are saved to the `./models/` directory by default.

### Test

1. Enter the path to a converted model (e.g., `./models/LFM2-1.2B-RAG-q4`)
2. Type a prompt or select an example
3. Adjust generation parameters (max tokens, temperature, etc.)
4. Click **Generate**

### Models

Browse previously converted models and see their sizes.

## Requirements

- macOS with Apple Silicon
- Python 3.9+
- ~2 GB free disk space per converted model (varies by model size and quantization)

## References

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) — Language model tools for MLX
- [Gradio](https://www.gradio.app/) — Web UI framework
