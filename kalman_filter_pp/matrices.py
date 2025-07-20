import numpy as np

#* Transition matrix for the Kalman filter
F = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 0, 1]])

#* Initial covariance matrix for the Kalman filter
#! CERN data values
P = np.array([[4e-2, 0e-0,   0e-0, 0e-0,   0e-0],    #* d0
              [0e-0, 6.7e-6, 0e-0, 0e-0,   0e-0],    #* phi0
              [0e-0, 0e-0,   2e-2, 0e-0,   0e-0],    #* q/pt
              [0e-0, 0e-0,   0e-0, 1.3e-0, 0e-0],    #* z0
              [0e-0, 0e-0,   0e-0, 0e-0,   7e-6]])   #* theta

#* Process noise covariance matrix for the Kalman filter
Q = np.array([[1e-0, 0e-0, 0e-0, 0e-0, 0e-0],   #* d1
              [0e-0, 1e-0, 0e-0, 0e-0, 0e-0],   #* phi0  
              [0e-0, 0e-0, 1e-0, 0e-0, 0e-0],   #* omega~
              [0e-0, 0e-0, 0e-0, 1e-0, 0e-0],   #* z0
              [0e-0, 0e-0, 0e-0, 0e-0, 1e-0]])  #* cotTheta

#* Measurement noise covariance for the Kalman filter
R = np.array([[1, 0],   #* phi
              [0, 1]])  #* z

def get_H(r : float, s : float) -> np.ndarray:
    """
    Get the Observation matrix H for the Kalman filter.

    Parameters
    -----------
    r (float): Radius of the helix.

    s (float): Arc length along the helix.

    Returns
    -----------
    np.ndarray: Observation matrix H.
    """

    #* phi = d1 / r + phi0 + r * omega~
    #* z = z0 + s * cotTheta
    #* where:
    #! Check if the following equations are correct:
    #* omega~ = 1 / (2 * (omega + d0))
    #* d1 = d0 * (1 - d0 * omega~)

    #*                d1    phi0  omega~  z0   cotTheta
    return np.array([[1/r,  1e-0, r,     0e-0, 0e-0],   #* phi
                     [0e-0, 0e-0, 0e-0,  1e-0,    s]])  #* z

