import numpy as np
from experiments.lambda_repair_eval import SCALES,selected


def test_full_support_finds_small_scale_well_missed_by_boundary_expansion():
    # A finite-scale crossover can beat the initial grid's small-scale edge,
    # even though a better local regime exists in the trained lower extension.
    curve=np.full((10,len(SCALES)),2.)
    curve[:,np.argmin(abs(SCALES-8))]=.5
    curve[:,0]=.1
    target=np.zeros(10)
    assert selected(curve,target)[0]==8
    assert selected(curve,target,'full_support_v1')[0]==1/256
