"""Small mathematical fixtures, not a substitute for the canonical archive.

Crescent follows the coefficient-space construction in pinned upstream
generators/utils/pca.py:crescent_moon_pca. No upstream module is imported into
the parent process and no PCA is fitted. NumPy RandomState preserves its draw
order; outputs are explicitly identified as regenerated pilot data.
"""

from __future__ import annotations

import numpy as np

PINNED_UPSTREAM_REVISION = "2dcb8e41015f53413ff1ddd049bb006c81a5df52"
PILOT_LIDS = {"rank3_gaussian_control": 3, "e7_crescent_moon_radius3.0": 3}


def sample_vp_fixture(name: str, count: int, seed: int) -> np.ndarray:
    if name not in PILOT_LIDS or count < 1:
        raise ValueError("unsupported VP fixture or invalid sample count")
    rng = np.random.RandomState(seed)
    result = np.zeros((count, 30), dtype=np.float32)
    if name == "rank3_gaussian_control":
        result[:, :3] = rng.normal(size=(count, 3))
        return result
    candidates = rng.uniform(-1.0, 1.0, size=(20 * count, 3))
    x, y = candidates[:, 0], candidates[:, 1]
    inside_outer = x**2 + y**2 <= 1.0
    outside_inner = np.sqrt(x**2 + (y - 0.1) ** 2) >= 0.899
    accepted = candidates[inside_outer & outside_inner]
    if len(accepted) < count:
        raise RuntimeError("crescent rejection sampler returned too few points")
    result[:, :3] = 3.0 * accepted[:count]
    return result
