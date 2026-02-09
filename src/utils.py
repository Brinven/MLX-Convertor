"""Helper utilities for the MLX Model Converter."""

import os
from pathlib import Path
from typing import Union


def get_model_size(path: Union[str, Path]) -> int:
    """Calculate total size of a model directory in bytes.

    Args:
        path: Path to the model directory.

    Returns:
        Total size in bytes.
    """
    total = 0
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string (e.g., '1.5 GB').
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def list_converted_models(output_dir: str = "./models") -> list:
    """List all converted models in the output directory.

    Args:
        output_dir: Directory containing converted models.

    Returns:
        List of dicts with keys: name, path, size.
    """
    models = []
    output_path = Path(output_dir)

    if not output_path.exists():
        return models

    for item in sorted(output_path.iterdir()):
        if item.is_dir():
            # Check if it looks like an MLX model (has config.json or *.safetensors)
            has_config = (item / "config.json").exists()
            has_weights = any(item.glob("*.safetensors")) or any(item.glob("*.npz"))

            if has_config or has_weights:
                size = get_model_size(item)
                models.append({
                    "name": item.name,
                    "path": str(item),
                    "size": format_size(size),
                })

    return models
