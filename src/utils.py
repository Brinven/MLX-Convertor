"""Helper utilities for the MLX Model Converter."""

import os
import shutil
import tempfile
import zipfile
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

    # Walk all subdirectories to find model folders at any depth
    for dirpath, dirnames, filenames in os.walk(output_path):
        item = Path(dirpath)
        has_config = "config.json" in filenames
        has_weights = any(f.endswith(".safetensors") or f.endswith(".npz") for f in filenames)

        if has_config or has_weights:
            size = get_model_size(item)
            models.append({
                "name": item.name,
                "path": str(item),
                "size": format_size(size),
            })
            # Don't descend into this model's subdirectories
            dirnames.clear()

    return sorted(models, key=lambda m: m["name"])


def zip_model(model_path: str) -> dict:
    """Zip a model directory for download/transfer.

    Args:
        model_path: Path to the model directory.

    Returns:
        Dict with keys: success, message, zip_path.
    """
    model_dir = Path(model_path)
    if not model_dir.exists() or not model_dir.is_dir():
        return {"success": False, "message": f"Model directory not found: {model_path}", "zip_path": ""}

    # Validate it looks like a model directory
    files = list(model_dir.iterdir())
    has_model_files = any(
        f.name == "config.json" or f.suffix in (".safetensors", ".npz")
        for f in files if f.is_file()
    )
    if not has_model_files:
        return {"success": False, "message": "Directory doesn't contain model files (config.json or weights).", "zip_path": ""}

    try:
        tmp_dir = tempfile.mkdtemp()
        zip_base = os.path.join(tmp_dir, model_dir.name)
        zip_path = shutil.make_archive(zip_base, "zip", root_dir=str(model_dir.parent), base_dir=model_dir.name)
        size_str = format_size(os.path.getsize(zip_path))
        return {"success": True, "message": f"Zipped {model_dir.name} ({size_str})", "zip_path": zip_path}
    except Exception as e:
        return {"success": False, "message": f"Failed to create zip: {e}", "zip_path": ""}


def import_model_zip(zip_path: str, output_dir: str = "./models") -> dict:
    """Import a model from an uploaded zip file.

    Args:
        zip_path: Path to the uploaded zip file.
        output_dir: Directory to extract the model into.

    Returns:
        Dict with keys: success, message, model_path.
    """
    if not zip_path or not os.path.isfile(zip_path):
        return {"success": False, "message": "No file provided.", "model_path": ""}

    if not zipfile.is_zipfile(zip_path):
        return {"success": False, "message": "File is not a valid zip archive.", "model_path": ""}

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

            # Validate: zip must contain config.json or weight files
            has_model_files = any(
                n.endswith("config.json") or n.endswith(".safetensors") or n.endswith(".npz")
                for n in names
            )
            if not has_model_files:
                return {
                    "success": False,
                    "message": "Invalid model zip. Archive must contain config.json or .safetensors files.",
                    "model_path": "",
                }

            # Determine the top-level directory name from the zip
            top_dirs = {n.split("/")[0] for n in names if "/" in n}

            os.makedirs(output_dir, exist_ok=True)

            if len(top_dirs) == 1:
                # Zip has a single top-level folder — extract directly
                model_name = top_dirs.pop()
                dest = os.path.join(output_dir, model_name)
                if os.path.exists(dest):
                    return {
                        "success": False,
                        "message": f"Model '{model_name}' already exists. Delete it first or rename.",
                        "model_path": "",
                    }
                zf.extractall(output_dir)
            else:
                # Flat zip or multiple top-level entries — extract into a folder named after the zip
                model_name = Path(zip_path).stem
                dest = os.path.join(output_dir, model_name)
                if os.path.exists(dest):
                    return {
                        "success": False,
                        "message": f"Model '{model_name}' already exists. Delete it first or rename.",
                        "model_path": "",
                    }
                os.makedirs(dest)
                zf.extractall(dest)

            size = get_model_size(dest)
            return {
                "success": True,
                "message": f"Imported '{model_name}' ({format_size(size)})",
                "model_path": dest,
            }

    except zipfile.BadZipFile:
        return {"success": False, "message": "Corrupted zip file.", "model_path": ""}
    except Exception as e:
        return {"success": False, "message": f"Import failed: {e}", "model_path": ""}
