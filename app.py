"""MLX Model Converter & Tester — A Gradio app for converting and testing MLX models."""

import json
import threading
from pathlib import Path

import gradio as gr

from src.converter import convert_model
from src.tester import generate_text, clear_cache
from src.utils import list_converted_models, zip_model, import_model_zip

# Load example prompts
EXAMPLES_PATH = Path(__file__).parent / "examples" / "example_prompts.json"
EXAMPLE_PROMPTS = {}
if EXAMPLES_PATH.exists():
    with open(EXAMPLES_PATH) as f:
        EXAMPLE_PROMPTS = json.load(f)

DEFAULT_OUTPUT_DIR = "./models"


# --- Conversion handler ---

def handle_convert(model_path, output_name, quantization, output_dir):
    """Handle the convert button click."""
    if not output_dir.strip():
        output_dir = DEFAULT_OUTPUT_DIR

    result = convert_model(
        model_path=model_path,
        output_name=output_name,
        quantization=quantization,
        output_dir=output_dir,
    )

    if result["success"]:
        return f"✅ {result['message']}"
    else:
        return f"❌ {result['message']}"


# --- Testing handler ---

def handle_generate(model_path, prompt, max_tokens, temperature, top_p, repetition_penalty):
    """Handle the generate button click."""
    result = generate_text(
        model_path=model_path,
        prompt=prompt,
        max_tokens=int(max_tokens),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    if result["success"]:
        return result["response"]
    else:
        return f"❌ {result['error']}"


def load_example_prompt(example_name):
    """Load an example prompt by name."""
    return EXAMPLE_PROMPTS.get(example_name, "")


def get_model_choices():
    """Get list of converted model paths for the dropdown."""
    models = list_converted_models(DEFAULT_OUTPUT_DIR)
    return [m["path"] for m in models]


def refresh_model_list():
    """Refresh the list of converted models."""
    models = list_converted_models(DEFAULT_OUTPUT_DIR)
    if not models:
        return "No converted models found in ./models/"
    lines = []
    for m in models:
        lines.append(f"• **{m['name']}** — {m['size']}\n  `{m['path']}`")
    return "\n\n".join(lines)


def get_model_names():
    """Get list of converted model names for the selector dropdown."""
    models = list_converted_models(DEFAULT_OUTPUT_DIR)
    return [m["name"] for m in models]


# --- Download/Upload handlers ---

def handle_download(model_name):
    """Handle the download button click."""
    if not model_name:
        return gr.update(), "❌ Please select a model to download."

    # Find the model path by name
    models = list_converted_models(DEFAULT_OUTPUT_DIR)
    match = next((m for m in models if m["name"] == model_name), None)
    if not match:
        return gr.update(), f"❌ Model '{model_name}' not found."

    result = zip_model(match["path"])
    if result["success"]:
        return result["zip_path"], f"✅ {result['message']}"
    else:
        return gr.update(), f"❌ {result['message']}"


def handle_upload(file):
    """Handle the upload/import of a model zip."""
    if file is None:
        return "❌ No file uploaded."

    file_path = file.name if hasattr(file, "name") else str(file)
    result = import_model_zip(file_path, DEFAULT_OUTPUT_DIR)
    if result["success"]:
        return f"✅ {result['message']}"
    else:
        return f"❌ {result['message']}"


# --- Build Gradio UI ---

def create_app():
    """Create and return the Gradio app."""
    with gr.Blocks(
        title="MLX Model Converter",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 0.5em; }
        .main-header h1 { margin-bottom: 0.1em; }
        .main-header p { color: #666; font-size: 0.95em; }
        """,
    ) as app:
        gr.HTML(
            """
            <div class="main-header">
                <h1>🧪 MLX Model Converter</h1>
                <p>Convert HuggingFace models to Apple MLX format and test them</p>
            </div>
            """
        )

        with gr.Tabs():
            # === Convert Tab ===
            with gr.Tab("🔄 Convert", id="convert"):
                gr.Markdown("### Convert a HuggingFace model to MLX format")

                with gr.Row():
                    with gr.Column(scale=2):
                        convert_model_path = gr.Textbox(
                            label="HuggingFace Model Path",
                            placeholder="e.g., LiquidAI/LFM2-1.2B-RAG",
                            info="The model identifier from huggingface.co",
                        )
                        convert_output_name = gr.Textbox(
                            label="Output Name (optional)",
                            placeholder="Leave blank for auto-generated name",
                            info="Name for the converted model directory",
                        )

                    with gr.Column(scale=1):
                        convert_quantization = gr.Dropdown(
                            choices=["4-bit", "8-bit", "bf16"],
                            value="4-bit",
                            label="Quantization",
                            info="4-bit: smallest, 8-bit: balanced, bf16: full precision",
                        )
                        convert_output_dir = gr.Textbox(
                            label="Output Directory",
                            value=DEFAULT_OUTPUT_DIR,
                            info="Where to save converted models",
                        )

                convert_btn = gr.Button("🚀 Convert Model", variant="primary", size="lg")
                convert_status = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    show_copy_button=True,
                )

                convert_btn.click(
                    fn=handle_convert,
                    inputs=[convert_model_path, convert_output_name, convert_quantization, convert_output_dir],
                    outputs=convert_status,
                )

            # === Test Tab ===
            with gr.Tab("💬 Test", id="test"):
                gr.Markdown("### Test a converted MLX model")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            test_model_path = gr.Dropdown(
                                choices=get_model_choices(),
                                label="Model",
                                allow_custom_value=True,
                                info="Select a converted model or type a custom path",
                                scale=4,
                            )
                            refresh_models_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)
                        test_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=5,
                        )

                        # Example prompt buttons
                        if EXAMPLE_PROMPTS:
                            gr.Markdown("**Example prompts:**")
                            with gr.Row():
                                for name in EXAMPLE_PROMPTS:
                                    btn = gr.Button(name.capitalize(), size="sm", variant="secondary")
                                    btn.click(
                                        fn=lambda n=name: EXAMPLE_PROMPTS[n],
                                        outputs=test_prompt,
                                    )

                    with gr.Column(scale=1):
                        test_max_tokens = gr.Slider(
                            minimum=50, maximum=2000, value=512, step=50,
                            label="Max Tokens",
                        )
                        test_temperature = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                            label="Temperature",
                            info="0 = deterministic, 1 = creative",
                        )
                        test_top_p = gr.Slider(
                            minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                            label="Top-p",
                        )
                        test_rep_penalty = gr.Slider(
                            minimum=1.0, maximum=1.5, value=1.0, step=0.05,
                            label="Repetition Penalty",
                        )

                with gr.Row():
                    generate_btn = gr.Button("▶️ Generate", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear Cache", size="lg")

                test_output = gr.Textbox(
                    label="Response",
                    lines=10,
                    interactive=False,
                    show_copy_button=True,
                )

                refresh_models_btn.click(
                    fn=lambda: gr.update(choices=get_model_choices()),
                    outputs=test_model_path,
                )
                generate_btn.click(
                    fn=handle_generate,
                    inputs=[test_model_path, test_prompt, test_max_tokens, test_temperature, test_top_p, test_rep_penalty],
                    outputs=test_output,
                )
                clear_btn.click(
                    fn=lambda: (clear_cache(), "Cache cleared.")[1],
                    outputs=test_output,
                )

            # === Models Tab ===
            with gr.Tab("📂 Models", id="models"):
                gr.Markdown("### Converted Models")
                models_display = gr.Markdown("Click Refresh to see converted models.")
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    model_selector = gr.Dropdown(
                        choices=get_model_names(),
                        label="Select Model",
                        scale=3,
                    )
                refresh_btn.click(
                    fn=lambda: (refresh_model_list(), gr.update(choices=get_model_names())),
                    outputs=[models_display, model_selector],
                )

                gr.Markdown("### Download Model")
                gr.Markdown("Download a model as a zip file for transfer to another machine.")
                download_btn = gr.Button("📥 Download as Zip", variant="primary")
                download_status = gr.Textbox(label="Status", lines=2, interactive=False)
                download_file = gr.File(label="Download", interactive=False)
                download_btn.click(
                    fn=handle_download,
                    inputs=model_selector,
                    outputs=[download_file, download_status],
                )

                gr.Markdown("### Import Model")
                gr.Markdown("Upload a model zip file (e.g., from Heretic Converter on your PC).")
                upload_file = gr.File(label="Upload Model Zip", file_types=[".zip"])
                import_status = gr.Textbox(label="Import Status", lines=2, interactive=False)
                upload_file.change(
                    fn=handle_upload,
                    inputs=upload_file,
                    outputs=import_status,
                )

            # === About Tab ===
            with gr.Tab("ℹ️ About", id="about"):
                gr.Markdown(
                    """
                    ### MLX Model Converter

                    A simple tool for converting HuggingFace models to Apple's MLX format
                    and testing them locally on Apple Silicon Macs.

                    **Requirements:**
                    - macOS with Apple Silicon (M1/M2/M3/M4)
                    - Python 3.9+

                    **Built with:**
                    - [Gradio](https://www.gradio.app/) — UI framework
                    - [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms) — MLX language model tools
                    - [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework

                    **Version:** 0.1.0
                    """
                )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", inbrowser=True)
