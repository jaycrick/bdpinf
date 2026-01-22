"""
Common utilities for API data collection and processing.
"""

import os
import urllib.request

import torch


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def download_file(url: str, local_path: str, overwrite: bool = False) -> None:
    """
    Download a file from a URL to a local path.

    Args:
        url: The URL to download from
        local_path: The local path to save to
        overwrite: Whether to overwrite existing files

    Raises:
        Exception: If download fails
    """
    if not overwrite and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return

    try:
        print(f"Downloading {url} to {local_path}")
        # Ensure directory exists
        ensure_directory(os.path.dirname(local_path))
        urllib.request.urlretrieve(url, local_path)
    except Exception as e:
        raise Exception(f"Error downloading {url}: {e}")


def safe_int(value: str | None) -> int | None:
    try:
        if value is not None:
            return int(value)
        else:
            return None
    except (TypeError, ValueError):
        return None


def ensure_1d_tensor(x: torch.Tensor | float | int) -> torch.Tensor:
    """
    Convert scalar or array-like input to a 1D PyTorch tensor.

    Parameters
    ----------
    x : torch.Tensor | float | int
        Input value to convert. Can be a scalar or tensor of any shape.

    Returns
    -------
    torch.Tensor
        1D tensor with dtype torch.float64. Scalars become shape (1,),
        multi-dimensional arrays are flattened.

    Examples
    --------
    >>> ensure_1d_tensor(5.0)
    tensor([5.], dtype=torch.float64)
    >>> ensure_1d_tensor(torch.tensor([[1, 2], [3, 4]]))
    tensor([1., 2., 3., 4.], dtype=torch.float64)
    """
    return torch.as_tensor(x, dtype=torch.float64).reshape(-1)
