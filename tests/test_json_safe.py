"""Tests for json_safe type-conversion utility."""

import jax.numpy as jnp
import numpy as np
import pandas as pd

from inf_bp_texts import utils


class TestJsonSafe:
    """Test json_safe utility."""

    def test_json_safe_conversions(self) -> None:
        assert utils.json_safe(np.int32(5)) == 5
        assert isinstance(utils.json_safe(np.int32(5)), int)
        assert utils.json_safe(np.float64(1.5)) == 1.5
        assert isinstance(utils.json_safe(np.float64(1.5)), float)
        assert isinstance(utils.json_safe(np.array([1, 2])), list)
        jax_arr = jnp.array([1.0, 2.0])
        out = utils.json_safe(jax_arr)
        assert isinstance(out, list) and out == [1.0, 2.0]
        ts = pd.Timestamp("2020-01-01T00:00:00Z")
        assert isinstance(utils.json_safe(ts), str)

    def test_json_safe_passthrough(self) -> None:
        # Already-serializable values (and None) are returned unchanged --
        # json.dumps never invokes default() for these, so json_safe doesn't
        # need an explicit branch for them.
        assert utils.json_safe(5) == 5
        assert utils.json_safe(1.5) == 1.5
        assert utils.json_safe(None) is None
        assert utils.json_safe("text") == "text"
