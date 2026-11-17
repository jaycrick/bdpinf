"""Common utilities for API data collection and processing."""

import operator
from collections.abc import Callable
from numbers import Integral, Real

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def format_ordinal(n: int) -> str:
    """Format an integer with its ordinal suffix (1st, 2nd, 3rd, 4th, 11th, ...).

    Examples
    --------
    >>> format_ordinal(2)
    '2nd'
    >>> format_ordinal(11)
    '11th'
    >>> format_ordinal(21)
    '21st'

    """
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
    return jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)).reshape(-1)


def ensure_1d_tensors(*values: object) -> tuple[jax.Array, ...]:
    """Convert multiple scalars or array-likes to 1D float64 JAX arrays."""
    return tuple(ensure_1d_tensor(v) for v in values)


def json_safe(value: object) -> object:
    """Convert numpy/JAX/pandas types not natively JSON-serializable to plain Python.

    Intended as a ``json.dumps(..., default=json_safe)`` hook: ``json`` already
    serializes native ``int``/``float``/``None``/etc. without calling ``default``,
    so only the numpy/JAX/pandas types below need explicit handling here.
    Anything else is returned unchanged.
    """
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    # JAX arrays
    if hasattr(value, "tolist") and hasattr(value, "device"):
        return np.array(value).tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _is_nan_like(value: object) -> bool:
    """Whether ``value != value`` holds (NaN, ``NaT``); False if it cannot be tested."""
    try:
        return bool(operator.ne(value, value))
    except Exception:  # noqa: BLE001  # arbitrary objects: __ne__ may raise anything
        return False


def _format_integer(value: Integral) -> object:
    """Integers below 1e4 stay integers; larger ones use scientific notation."""
    integer_value = int(value)
    if abs(integer_value) >= 1e4:
        return f"{float(integer_value):.2e}"
    return integer_value


def _format_real(value: Real) -> object:
    """Format a real number: non-finite unchanged, integral as ``int``, else text."""
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


def format_table_value(value: object) -> object:
    """Format a scalar for display in interactive tables."""
    if (
        value is None
        or isinstance(value, (str, bytes))
        or _is_nan_like(value)
        or isinstance(value, (bool, np.bool_))
    ):
        return value
    if isinstance(value, Integral):
        return _format_integer(value)
    if isinstance(value, Real):
        return _format_real(value)
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


def derive_worker_seeds(base_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent integer seeds from a single base seed.

    Uses ``jax.random.key`` + ``jax.random.split`` to generate statistically
    independent subkeys, then draws one integer per subkey. Prefer this over ad
    hoc integer arithmetic (e.g. ``base_seed + i``) when seeding independent
    ``numpy.random.default_rng`` streams across parallel (joblib) workers ---
    sequential-integer seeds rely on the downstream RNG's seed-mixing to
    decorrelate adjacent streams, whereas ``jax.random.split`` gives an explicit
    independence guarantee per stream.

    Parameters
    ----------
    base_seed : int
        Seed for the top-level ``jax.random.key``.
    n : int
        Number of independent seeds to derive.

    Returns
    -------
    list[int]
        ``n`` plain Python ints in ``[0, 2**31 - 1)``, suitable for
        ``numpy.random.default_rng`` or any ``random_seed: int`` parameter.

    """

    def _draw_one(subkey: jax.Array) -> jax.Array:
        return jax.random.randint(subkey, (), 0, 2**31 - 1)

    top_key = jax.random.key(base_seed)
    subkeys = jax.random.split(top_key, n)
    draws = jax.vmap(_draw_one)(subkeys)
    return [int(s) for s in draws]
