"""Common utilities for API data collection and processing."""

from collections.abc import Callable
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
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


def json_safe(value: object) -> object:
    """Convert common numpy/torch/pandas types to JSON-serializable Python types."""
    try:
        import numpy as _np
        import pandas as _pd
        import torch as _torch
    except Exception:
        return value

    if isinstance(value, (_np.integer, int)):
        return int(value)
    if isinstance(value, (_np.floating, float)):
        return float(value)
    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, _torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (_pd.Timestamp,)):
        return value.isoformat()
    if value is None:
        return None
    return value


def format_table_value(value: object) -> object:
    """Format a scalar for display in interactive tables."""
    if value is None:
        return None

    if isinstance(value, (str, bytes)):
        return value

    try:
        if value is None or value != value:
            return value
    except Exception:
        pass

    if isinstance(value, (bool, np.bool_)):
        return value

    if isinstance(value, Integral):
        integer_value = int(value)
        if abs(integer_value) >= 1e4:
            return f"{float(integer_value):.2e}"
        return integer_value

    if isinstance(value, Real):
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            return value

        if abs(numeric_value) >= 1e4:
            return f"{numeric_value:.2e}"

        if numeric_value.is_integer():
            return int(numeric_value)

        magnitude = abs(numeric_value)
        if 1e-1 <= magnitude < 1e4:
            return f"{numeric_value:.2f}"
        return f"{numeric_value:.2e}"

    return value


def format_table_dataframe(
    dataframe: pd.DataFrame,
    formatter: Callable[[object], object] = format_table_value,
) -> pd.DataFrame:
    """Return a display copy of a dataframe with formatted table values."""
    return pd.DataFrame(
        {
            column_name: pd.Series(
                [formatter(value) for value in dataframe[column_name].tolist()],
                index=dataframe.index,
                dtype=object,
            )
            for column_name in dataframe.columns
        },
        index=dataframe.index,
    )


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
