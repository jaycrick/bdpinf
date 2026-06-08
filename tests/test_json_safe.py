"""Tests for json_safe type-conversion utility."""

import numpy as np
import pandas as pd
import torch

from inf_bp_texts import utils


class TestJsonSafe:
    """Test json_safe utility."""

    def test_json_safe_conversions(self) -> None:
        assert utils.json_safe(np.int32(5)) == 5
        assert isinstance(utils.json_safe(np.array([1, 2])), list)
        t = torch.tensor([1.0, 2.0])
        out = utils.json_safe(t)
        assert isinstance(out, list) and out == [1.0, 2.0]
        ts = pd.Timestamp("2020-01-01T00:00:00Z")
        assert isinstance(utils.json_safe(ts), str)
