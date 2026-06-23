from __future__ import annotations

import pytest

import numpy as np

from onelife.duc_wm.mujoco_ext import CONTEXT_NAMES, enabled_context_names
from onelife.duc_wm.planning_eval import align_raw_context_to_templates
from onelife.duc_wm.templates import default_mujoco_templates


def test_enabled_context_names_validates_variant_tokens():
    assert enabled_context_names("all") == set(CONTEXT_NAMES)
    assert enabled_context_names("none") == set()
    assert enabled_context_names(" wind + sticky ") == {"wind", "sticky"}

    with pytest.raises(ValueError, match="unknown MuJoCo extension context"):
        enabled_context_names("wind+bad_context")


def test_align_raw_context_to_templates_for_oracle_planning():
    templates = default_mujoco_templates("Swimmer-v5", state_dim=8, action_dim=2)
    raw = np.zeros(len(CONTEXT_NAMES), dtype=np.float32)
    raw[CONTEXT_NAMES.index("wind")] = 0.25
    raw[CONTEXT_NAMES.index("friction")] = -0.1

    aligned = align_raw_context_to_templates(raw, templates)
    name_to_index = {template.name: index for index, template in enumerate(templates)}

    assert aligned.shape == (len(templates),)
    assert aligned[name_to_index["actuation"]] == 1.0
    assert aligned[name_to_index["wind"]] == pytest.approx(0.25)
    assert aligned[name_to_index["friction"]] == pytest.approx(-0.1)
    assert aligned[name_to_index["unknown"]] == 0.0
