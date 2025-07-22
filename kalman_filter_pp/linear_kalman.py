import numpy as np

from .logger import get_logger

logger = get_logger()

def kf_predict(x : np.ndarray, P : np.ndarray, F : np.ndarray, Q : np.ndarray) -> tuple:
    """
    Predict the next state and covariance using the Kalman filter prediction step.

    Parameters
    -----------
    x (np.ndarray): Current state estimate.

    P (np.ndarray): Current estimate covariance.

    F (np.ndarray): State transition matrix.

    Q (np.ndarray): Process noise covariance.

    Returns
    -----------
    tuple: Predicted state and covariance.

    Notes
    -----------
    This function implements the Kalman filter prediction step. It computes the predicted state and covariance
    based on the current state, covariance, state transition matrix, and process noise covariance.

    Example usage
    -----------
    >>> x = np.array([[1], [2]])
    >>> P = np.array([[1, 0], [0, 1]])
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.array([[0.1, 0], [0, 0.1]])
    >>> x_pred, P_pred = kf_predict(x, P, F, Q)
    >>> print(x_pred)
    >>> print(P_pred)
    Output:
    [[3]
     [2]]
    [[2.1 1. ]
     [1.  1.1]]
    """

    x_pred = F @ x.T
    P_pred = F @ P @ F.T + Q

    logger.debug(f"Predicted state: {x_pred}")

    return x_pred.T, P_pred

def kf_update(x_pred : np.ndarray, P_pred : np.ndarray, z : np.ndarray, H : np.ndarray, R : np.ndarray) -> tuple:
    """
    Update the state estimate and covariance using the Kalman filter update step.

    Parameters
    -----------
    x_pred (np.ndarray): Predicted state estimate.

    P_pred (np.ndarray): Predicted estimate covariance.

    z (np.ndarray): Measurement vector.

    H (np.ndarray): Observation matrix.

    R (np.ndarray): Measurement noise covariance.

    Returns
    -----------
    tuple: Updated state, updated covariance, and Kalman gain.

    Notes
    -----------
    This function implements the Kalman filter update step using the Joseph form for numerical stability.

    Example usage
    -----------
    >>> x_pred = np.array([[1], [2]])
    >>> P_pred = np.array([[1, 0], [0, 1]])
    >>> z = np.array([[1.5], [2.5]])
    >>> H = np.array([[1, 0], [0, 1]])
    >>> R = np.array([[0.1, 0], [0, 0.1]])
    >>> x_updated, P_updated, K = kf_update(x_pred, P_pred, z, H, R)
    >>> print(x_updated)
    >>> print(P_updated)
    >>> print(K)
    Output:
    [[1.45454545]
     [2.45454545]]
    [[0.09090909 0.        ]
     [0.         0.09090909]]
    [[0.90909091 0.        ]
     [0.         0.90909091]]
    """
    
    y = z - H @ x_pred                      #* Innovation
    S = H @ P_pred @ H.T + R                #* Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)     #* Kalman gain

    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(P_pred)) - K @ H) @ P_pred @ (np.eye(len(P_pred)) - K @ H).T + K @ R @ K.T #* Joseph formula | numerical stable

    # logger.debug(f"Updated state: {x_updated}")
    # logger.debug(f"Updated covariance: {P_updated}")
    # logger.debug(f"Kalman gain: {K}")

    return x_updated, P_updated, K