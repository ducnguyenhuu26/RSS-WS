from __future__ import annotations

import pytest

from onelife.duc_wm.mujoco_ext import CONTEXT_NAMES, enabled_context_names


def test_enabled_context_names_validates_variant_tokens():
    assert enabled_context_names("all") == set(CONTEXT_NAMES)
    assert enabled_context_names("none") == set()
    assert enabled_context_names(" wind + sticky ") == {"wind", "sticky"}

    with pytest.raises(ValueError, match="unknown MuJoCo extension context"):
        enabled_context_names("wind+bad_context")
