"""Model testing/inference logic for MLX models."""

from pathlib import Path

# Simple model cache to avoid reloading
_model_cache = {
    "path": None,
    "model": None,
    "tokenizer": None,
}


def load_model(model_path: str) -> tuple:
    """Load an MLX model and tokenizer, with caching.

    Args:
        model_path: Path to the MLX model directory.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        FileNotFoundError: If the model path doesn't exist.
        RuntimeError: If the model fails to load.
    """
    # Return cached model if path matches
    if _model_cache["path"] == model_path and _model_cache["model"] is not None:
        return _model_cache["model"], _model_cache["tokenizer"]

    # Validate path exists
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if not path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")

    try:
        from mlx_lm import load

        model, tokenizer = load(model_path)

        # Cache the loaded model
        _model_cache["path"] = model_path
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer

        return model, tokenizer

    except ImportError:
        raise RuntimeError("mlx-lm is not installed. Run: pip install mlx-lm")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def generate_text(
    model_path: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> dict:
    """Generate text using a loaded MLX model.

    Args:
        model_path: Path to the MLX model directory.
        prompt: Input prompt for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
        top_p: Top-p sampling parameter.
        repetition_penalty: Penalty for repeating tokens.

    Returns:
        Dict with keys: success, response, error.
    """
    if not prompt or not prompt.strip():
        return {"success": False, "response": "", "error": "Prompt cannot be empty."}

    if not model_path or not model_path.strip():
        return {"success": False, "response": "", "error": "Please provide a model path."}

    try:
        model, tokenizer = load_model(model_path)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        return {"success": False, "response": "", "error": str(e)}

    try:
        from mlx_lm import generate

        kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
        }

        response = generate(**kwargs)

        return {"success": True, "response": response, "error": ""}

    except Exception as e:
        return {"success": False, "response": "", "error": f"Generation failed: {e}"}


def clear_cache():
    """Clear the cached model to free memory."""
    _model_cache["path"] = None
    _model_cache["model"] = None
    _model_cache["tokenizer"] = None
