"""Common utilities for API data collection and processing."""

from pathlib import Path

import torch


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_int(value: str | None) -> int | None:
    try:
        if value is not None:
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def ensure_1d_tensor(x: torch.Tensor | float) -> torch.Tensor:
    """Convert scalar or array-like input to a 1D PyTorch tensor.

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


def ensure_1d_tensors(*values: torch.Tensor | float) -> tuple[torch.Tensor, ...]:
    return tuple(ensure_1d_tensor(value) for value in values)


def validate_matching_shapes(
    *tensors: torch.Tensor,
    names: tuple[str, ...] | None = None,
) -> None:
    if len(tensors) < 2:
        return
    if names is not None and len(names) != len(tensors):
        msg = (
            "names must be the same length as tensors when provided. "
            f"Got names={len(names)}, tensors={len(tensors)}"
        )
        raise ValueError(
            msg,
        )

    reference_shape = tensors[0].shape
    mismatches = [
        (idx, tensor.shape)
        for idx, tensor in enumerate(tensors)
        if tensor.shape != reference_shape
    ]
    if mismatches:
        parts = []
        for idx, shape in mismatches:
            label = names[idx] if names is not None else f"tensor_{idx}"
            parts.append(f"{label} shape={shape}")
        reference_label = names[0] if names is not None else "tensor_0"
        message = (
            "Tensors must have matching shapes. "
            f"Reference {reference_label} shape={reference_shape}. "
            "Mismatches: " + ", ".join(parts)
        )
        raise ValueError(message)


def validate_matching_lengths(
    *tensors: torch.Tensor,
    names: tuple[str, ...] | None = None,
) -> None:
    if len(tensors) < 2:
        return
    if names is not None and len(names) != len(tensors):
        msg = (
            "names must be the same length as tensors when provided. "
            f"Got names={len(names)}, tensors={len(tensors)}"
        )
        raise ValueError(
            msg,
        )

    reference_len = len(tensors[0])
    mismatches = [
        (idx, len(tensor))
        for idx, tensor in enumerate(tensors)
        if len(tensor) != reference_len
    ]
    if mismatches:
        parts = []
        for idx, length in mismatches:
            label = names[idx] if names is not None else f"tensor_{idx}"
            parts.append(f"{label} length={length}")
        reference_label = names[0] if names is not None else "tensor_0"
        message = (
            "Tensors must have matching lengths. "
            f"Reference {reference_label} length={reference_len}. "
            "Mismatches: " + ", ".join(parts)
        )
        raise ValueError(message)
