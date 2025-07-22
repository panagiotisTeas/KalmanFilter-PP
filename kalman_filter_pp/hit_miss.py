import numpy as np

from scipy.stats import chi2

from .logger import get_logger
from .jacobians import rz_jacobian, xy_jacobian
from .helix import Helix

def is_inside_error_ellipse(measured : np.ndarray, predicted : np.ndarray, cov_helix : np.ndarray, jacobian : np.ndarray, confidence_level : float = 0.95) -> bool:

    cov_2d = jacobian @ cov_helix @ jacobian.T
    delta = measured - predicted

    try:
        cov_inv = np.linalg.inv(cov_2d)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_2d)

    d2 = delta.T @ cov_inv @ delta
    threshold = chi2.ppf(confidence_level, df=2)
    inside = d2 <= threshold

    return inside

def hit_miss(measured : np.ndarray, predicted : np.ndarray, helix : Helix, cov_helix : np.ndarray, volume : int, arclength : float, confidence_level : float = 0.95) -> bool:
    
    if volume in [8, 13, 17]:
        jacobian = rz_jacobian(helix, np.sqrt(predicted[0]**2 + predicted[1]**2), np.atan2(predicted[1], predicted[0]), arclength)
        return is_inside_error_ellipse(np.array([np.atan2(measured[1], measured[0]), measured[2]]), np.array([np.atan2(predicted[1], predicted[0]), predicted[2]]), cov_helix, jacobian, confidence_level)
    else:
        jacobian = xy_jacobian(helix, predicted[0], predicted[1], arclength)
        return is_inside_error_ellipse(measured[:2], predicted[:2], cov_helix, jacobian, confidence_level)
