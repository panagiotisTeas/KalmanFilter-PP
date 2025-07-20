import numpy as np

from .helix import Helix

#* These jacobians are straight from Applied Fitting Theory IV | Formulas for Track Fitting by Paul Avery
#! Must be checked against the original paper for correctness.

def xy_jacobian(helix: Helix, x: float, y: float, s : float) -> np.ndarray:
    pass

def rz_jacobian(helix: Helix, r: float, phi : float, s: float) -> np.ndarray:
    pass
