import numpy as np

from .helix import Helix

#* These jacobians are straight from Applied Fitting Theory IV | Formulas for Track Fitting by Paul Avery
#! Must be checked against the original paper for correctness.

def xy_jacobian(helix: Helix, x: float, y: float, s : float) -> np.ndarray:
    """
    Projects the helix state to a plane at coordinates (x, y). (Detector endcap surface)

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    x (float): The x-coordinate of the plane.

    y (float): The y-coordinate of the plane.

    s (float): The arc length along the helix to the point of interest.

    Returns
    -------
    np.ndarray: The Jacobian matrix for the xy projection.

    The Jacobian matrix is defined as:
    [dx/dd0, dx/dphi0, dx/domega, dx/dz0, dx/dcotTheta]
    [dy/dd0, dy/dphi0, dy/domega, dy/dz0, dy/dcotTheta]
    """
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()

    dx_dd0 = - np.sin(phi0)
    dx_dphi0 = - y
    dx_domega = (np.sin(phi0) - np.sin(omega * s + phi0) + omega * s * np.cos(omega * s + phi0)) / omega**2
    dx_dz0 = - np.cos(omega * s + phi0) / cotTheta
    dx_dcotTheta = - np.cos(omega * s + phi0) * s / cotTheta
    dy_dd0 = np.cos(phi0)
    dy_dphi0 = x
    dy_domega = (- np.cos(phi0) + np.cos(omega * s + phi0) + omega * s * np.sin(omega * s + phi0)) / omega**2
    dy_dz0 = - np.sin(omega * s + phi0) / cotTheta
    dy_dcotTheta = - np.sin(omega * s + phi0) * s / cotTheta

    return np.array([[dx_dd0, dx_dphi0, dx_domega, dx_dz0, dx_dcotTheta],
                     [dy_dd0, dy_dphi0, dy_domega, dy_dz0, dy_dcotTheta]])

def rz_jacobian(helix: Helix, r: float, phi : float, s: float) -> np.ndarray:
    """
    Projects the helix state to a cylindrical surface at radius r and angle phi. (Detector barrel surface)

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    r (float): The radius of the cylindrical surface.

    phi (float): The angle at which the cylindrical surface is located.

    s (float): The arc length along the helix to the point of interest.

    Returns
    -------
    np.ndarray: The Jacobian matrix for the cylindrical projection.

    The Jacobian matrix is defined as:
    [dphi/dd0, dphi/dphi0, dphi/domega, dphi/dz0, dphi/dcotTheta]
    [dz/dd0,   dz/dphi0,   dz/domega,   dz/dz0,   dz/dcotTheta]
    """
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()

    dphi_dd0 = (1 / (r * np.cos(phi - phi0))) * (1 + omega * d0 - omega**2 * (r**2 - d0**2) / 2) / (1 + omega * d0)**2
    dphi_dphi0 = 1
    dphi_domega = (1 / (r * np.cos(phi - phi0))) * (r**2 - d0**2) / (1 + omega * d0)**2
    dphi_dz0 = 0
    dphi_dcotTheta = 0
    dz_dd0 = - 2 * s / omega - 2 * np.sin(omega * s / 2) * (d0 / (r**2 - d0**2) + omega / (2 + 2 * omega * d0)) / omega
    dz_dphi0 = 0
    dz_domega = 4 * (1 + omega * d0 / 2) * np.sin(omega * s / 2) / ((1 + omega * d0) * (omega**2))
    dz_dz0 = 1
    dz_dcotTheta = s

    return np.array([[dphi_dd0, dphi_dphi0, dphi_domega, dphi_dz0, dphi_dcotTheta],
                     [dz_dd0,   dz_dphi0,   dz_domega,   dz_dz0,   dz_dcotTheta]])

#* These Jacobians were calculated by hand (not from the paper).
#! Must be checked for correctness.

def forward_jacobian(helix: Helix) -> np.ndarray:
    """
    Forward Jacobian
    [d0, phi0, omega, z0, theta] -> [d1, phi0, omega~, z0, cotTheta]

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    Returns
    -------
    np.ndarray: The forward Jacobian matrix.
    """
    
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()
    a = omega * d0
    b = 1 + omega * d0

    return np.array([[(1/2)*(a/b)**2 - (a/b) + 1, 0, (1/2)*(a*d0**2)/(b**2) - (d0**2)/(2*b), 0, 0],
                     [0,                          1, 0,                                      0, 0],
                     [-(1/2)*omega**2/(b**2),     0, (1/2)*a/(b**2)+1/(2*b),                 0, 0],
                     [0,                          0, 0,                                      1, 0],
                     [0,                          0, 0,                                      0, -cotTheta**2-1]])

def backward_jacobian(helix: Helix) -> np.ndarray:
    """
    Backward Jacobian
    [d1, phi0, omega~, z0, cotTheta] -> [d0, phi0, omega, z0, theta]

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    Returns
    -------
    np.ndarray: The backward Jacobian matrix.
    """
    
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()
    a = omega * d0
    b = 1 + omega * d0

    return np.array([[b,          0, b*d0**2,           0, 0],
                     [0,          1, 0,                 0, 0],
                     [b*omega**2, 0, a**3+3*a**2+4*a+2, 0, 0],
                     [0,          0, 0,                 1, 0],
                     [0,          0, 0,                 0, 1/(-cotTheta**2-1)]])

def qpt_jacobian(helix: Helix) -> np.ndarray:
    """
    Jacobian for the q/pt parameter.
    [d0, phi0, q/pt, z0, theta] -> [d0, phi0, omega, z0, theta]

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    Returns
    -------
    np.ndarray: The Jacobian matrix for the q/pt parameter.
    """
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()
    
    return np.array([[1, 0, 0,      0, 0],
                     [0, 1, 0,      0, 0],
                     [0, 0, 0.6e-3, 0, 0],
                     [0, 0, 0,      1, 0],
                     [0, 0, 0,      0, 1]])

def omega_jacobian(helix: Helix) -> np.ndarray:
    """
    Jacobian for the omega parameter.
    [d0, phi0, omega, z0, theta] -> [d0, phi0, q/pt, z0, theta]

    Parameters
    ----------
    helix (Helix): The helix object containing the track parameters.

    Returns
    -------
    np.ndarray: The Jacobian matrix for the omega parameter.
    """
    
    d0, phi0, omega, z0, cotTheta = helix.get_state()
    
    return np.array([[1, 0, 0,              0, 0],
                     [0, 1, 0,              0, 0],
                     [0, 0, (0.6e-3)**(-1), 0, 0],
                     [0, 0, 0,              1, 0],
                     [0, 0, 0,              0, 1]])