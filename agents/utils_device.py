import torch


def get_device():
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_cuda_or_explain() -> None:
    """Print a helpful hint when CUDA is unavailable."""
    if not torch.cuda.is_available():
        print(
            "CUDA not available. This project is optimized for GPU. "
            "Run in Google Colab: Runtime → Change runtime type → GPU."
        )
