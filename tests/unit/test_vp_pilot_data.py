import hashlib

import numpy as np

from datasets.vp_pilot import sample_vp_fixture


def test_crescent_matches_frozen_pre_refactor_input():
    data = sample_vp_fixture("e7_crescent_moon_radius3.0", 256, 17)
    assert hashlib.sha256(data.tobytes()).hexdigest() == (
        "faf6f97d0ac9d862113591f0d5bb4dd3c26e75cc4e77de4bc14b6b0ecddd17cd"
    )
    assert data.shape == (256, 30)
    assert data.dtype == np.float32
    assert np.all(data[:, 3:] == 0)


def test_sampling_does_not_change_global_numpy_rng():
    np.random.seed(91)
    expected = np.random.randn(4)
    np.random.seed(91)
    sample_vp_fixture("rank3_gaussian_control", 16, 3)
    np.testing.assert_array_equal(np.random.randn(4), expected)
