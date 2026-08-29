from __future__ import annotations

from collections import Counter

from experiments import nf_readout_followup


def test_expected_extension_matrix_keeps_e1_spiral_in_known_lid() -> None:
    keys = nf_readout_followup._expected_extension_row_keys()

    assert len(keys) == 234
    assert Counter(key[1] for key in keys) == {
        "known_lid": 114,
        "e1_sample_size_stability": 78,
        "e5_paired_delta": 42,
    }
    spiral = [key for key in keys if key[3] == "e1_spiral_pca"]
    sampled = [key for key in keys if key[3].startswith("e1_sampled_fmnist")]
    assert spiral and {key[1] for key in spiral} == {"known_lid"}
    assert sampled and {key[1] for key in sampled} == {"e1_sample_size_stability"}
