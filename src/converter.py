"""Model conversion logic for converting HuggingFace models to MLX format."""

import os
import shutil
from pathlib import Path


def validate_model_path(model_path: str) -> tuple[bool, str]:
    """Validate a HuggingFace model path format.

    Args:
        model_path: HuggingFace model identifier (e.g., 'org/model-name').

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not model_path or not model_path.strip():
        return False, "Model path cannot be empty."

    model_path = model_path.strip()

    if "/" not in model_path:
        return False, "Model path should be in format 'org/model-name' (e.g., 'LiquidAI/LFM2-1.2B-RAG')."

    parts = model_path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return False, "Model path should have exactly one '/' separating org and model name."

    return True, ""


def get_output_path(output_name: str, output_dir: str = "./models") -> Path:
    """Build the output directory path for a converted model.

    Args:
        output_name: Name for the converted model directory.
        output_dir: Parent directory for converted models.

    Returns:
        Full path to the output model directory.
    """
    return Path(output_dir) / output_name


def convert_model(
    model_path: str,
    output_name: str = "",
    quantization: str = "4-bit",
    output_dir: str = "./models",
) -> dict:
    """Convert a HuggingFace model to MLX format.

    Args:
        model_path: HuggingFace model identifier.
        output_name: Name for the converted model. Defaults to model name with quantization suffix.
        quantization: One of '4-bit', '8-bit', or 'bf16'.
        output_dir: Directory to save converted models.

    Returns:
        Dict with keys: success, message, output_path, size.
    """
    # Validate input
    is_valid, error = validate_model_path(model_path)
    if not is_valid:
        return {"success": False, "message": error, "output_path": "", "size": ""}

    # Map quantization option to mlx-lm format
    quant_map = {
        "4-bit": "q4",
        "8-bit": "q8",
        "bf16": None,
    }
    quant = quant_map.get(quantization)
    if quantization not in quant_map:
        return {
            "success": False,
            "message": f"Invalid quantization option: {quantization}. Choose from: 4-bit, 8-bit, bf16.",
            "output_path": "",
            "size": "",
        }

    # Build output path
    if not output_name.strip():
        model_short_name = model_path.split("/")[-1]
        suffix = f"-{quant_map[quantization]}" if quant else "-bf16"
        output_name = f"{model_short_name}{suffix}"

    output_path = get_output_path(output_name, output_dir)

    # Check if output already exists
    if output_path.exists():
        return {
            "success": False,
            "message": f"Output directory already exists: {output_path}\nPlease choose a different name or delete the existing directory.",
            "output_path": str(output_path),
            "size": "",
        }

    # Check disk space (rough estimate: need at least 2GB free)
    try:
        free_space = shutil.disk_usage(output_dir if os.path.exists(output_dir) else ".").free
        if free_space < 2 * 1024 * 1024 * 1024:
            return {
                "success": False,
                "message": f"Low disk space: only {free_space / (1024**3):.1f} GB free. Models typically require several GB.",
                "output_path": "",
                "size": "",
            }
    except OSError:
        pass  # Skip disk check if it fails

    # Ensure output directory parent exists
    os.makedirs(output_dir, exist_ok=True)

    # Run conversion
    try:
        from mlx_lm import convert

        convert_args = {
            "hf_path": model_path,
            "mlx_path": str(output_path),
        }
        if quant is not None:
            convert_args["quantize"] = True
            convert_args["q_bits"] = int(quant[1])  # "q4" -> 4, "q8" -> 8

        convert(**convert_args)

        # Calculate output size
        from src.utils import get_model_size, format_size

        size = get_model_size(output_path)
        size_str = format_size(size)

        return {
            "success": True,
            "message": f"Model converted successfully!\n\nOutput: {output_path}\nSize: {size_str}\nQuantization: {quantization}",
            "output_path": str(output_path),
            "size": size_str,
        }

    except ImportError:
        return {
            "success": False,
            "message": "mlx-lm is not installed. Run: pip install mlx-lm",
            "output_path": "",
            "size": "",
        }
    except Exception as e:
        error_msg = str(e)

        # Provide friendly error messages for common issues
        if "404" in error_msg or "not found" in error_msg.lower():
            friendly = f"Model '{model_path}' was not found on HuggingFace. Please check the model path."
        elif "connection" in error_msg.lower() or "network" in error_msg.lower():
            friendly = "Network error. Please check your internet connection and try again."
        elif "disk" in error_msg.lower() or "space" in error_msg.lower() or "no space" in error_msg.lower():
            friendly = "Insufficient disk space. Please free up space and try again."
        else:
            friendly = f"Conversion failed: {error_msg}"

        # Clean up partial output on failure
        if output_path.exists():
            try:
                shutil.rmtree(output_path)
            except OSError:
                pass

        return {
            "success": False,
            "message": friendly,
            "output_path": "",
            "size": "",
        }
