"""Common utilities for API data collection and processing."""

from collections.abc import Callable
from numbers import Integral, Real
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def ensure_directory(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_int(value: str | None) -> int | None:
    """Convert a string to int, returning None on failure or None input."""
    try:
        if value is not None:
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def ensure_1d_tensor(x: object) -> jax.Array:
    """Convert scalar or array-like input to a 1D float64 JAX array.

    Parameters
    ----------
    x : array-like
        Input value to convert. Can be a scalar, numpy array, or JAX array.
        Multi-dimensional arrays are flattened.

    Returns
    -------
    jax.Array
        1D float64 JAX array. Scalars become shape (1,).

    Examples
    --------
    >>> ensure_1d_tensor(5.0)
    Array([5.], dtype=float64)
    >>> ensure_1d_tensor(np.array([[1, 2], [3, 4]]))
    Array([1., 2., 3., 4.], dtype=float64)

    """
    # Torch tensors expose __array__ but must be detached first
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()  # type: ignore[union-attr]
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)).reshape(-1)


def ensure_1d_tensors(*values: object) -> tuple[jax.Array, ...]:
    """Convert multiple scalars or array-likes to 1D float64 JAX arrays."""
    return tuple(ensure_1d_tensor(v) for v in values)


def require_tensors(**named: object) -> None:
    """No-op kept for API compatibility.

    In the JAX backend all inputs are normalised at the module boundary;
    this guard is no longer needed but is retained so call sites that pass
    ``_skip_normalization=True`` do not have to change.
    """


def json_safe(value: object) -> object:
    """Convert common numpy/JAX/pandas types to JSON-serializable Python types."""
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    # JAX arrays
    if hasattr(value, "tolist") and hasattr(value, "device"):
        return np.array(value).tolist()  # type: ignore[arg-type]
    # Torch tensors (if still present during transition)
    if hasattr(value, "detach") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()  # type: ignore[union-attr]
    if isinstance(value, pd.Timestamp):
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
    *arrays: jax.Array,
    names: tuple[str, ...] | None = None,
) -> None:
    """Raise ValueError if any array's shape differs from the first array's shape."""
    if len(arrays) < 2:
        return
    if names is not None and len(names) != len(arrays):
        msg = (
            "names must be the same length as arrays when provided. "
            f"Got names={len(names)}, arrays={len(arrays)}"
        )
        raise ValueError(msg)

    reference_shape = arrays[0].shape
    mismatches = [
        (idx, arr.shape)
        for idx, arr in enumerate(arrays)
        if arr.shape != reference_shape
    ]
    if mismatches:
        parts = []
        for idx, shape in mismatches:
            label = names[idx] if names is not None else f"array_{idx}"
            parts.append(f"{label} shape={shape}")
        reference_label = names[0] if names is not None else "array_0"
        message = (
            "Arrays must have matching shapes. "
            f"Reference {reference_label} shape={reference_shape}. "
            "Mismatches: " + ", ".join(parts)
        )
        raise ValueError(message)


def validate_matching_lengths(
    *arrays: jax.Array,
    names: tuple[str, ...] | None = None,
) -> None:
    """Raise ValueError if any array's length differs from the first array's."""
    if len(arrays) < 2:
        return
    if names is not None and len(names) != len(arrays):
        msg = (
            "names must be the same length as arrays when provided. "
            f"Got names={len(names)}, arrays={len(arrays)}"
        )
        raise ValueError(msg)

    reference_len = len(arrays[0])
    mismatches = [
        (idx, len(arr)) for idx, arr in enumerate(arrays) if len(arr) != reference_len
    ]
    if mismatches:
        parts = []
        for idx, length in mismatches:
            label = names[idx] if names is not None else f"array_{idx}"
            parts.append(f"{label} length={length}")
        reference_label = names[0] if names is not None else "array_0"
        message = (
            "Arrays must have matching lengths. "
            f"Reference {reference_label} length={reference_len}. "
            "Mismatches: " + ", ".join(parts)
        )
        raise ValueError(message)
