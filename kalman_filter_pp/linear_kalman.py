import numpy as np

def kf_predict(x : np.ndarray, P : np.ndarray, F : np.ndarray, Q : np.ndarray) -> tuple:
    """
    Predict the next state and covariance using the Kalman filter prediction step.

    Parameters:
    x (np.ndarray): Current state estimate.
    P (np.ndarray): Current estimate covariance.
    F (np.ndarray): State transition matrix.
    Q (np.ndarray): Process noise covariance.

    Returns:
    tuple: Predicted state and covariance.
    """

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred

def kf_update(x_pred : np.ndarray, P_pred : np.ndarray, z : np.ndarray, H : np.ndarray, R : np.ndarray) -> tuple:
    """
    Update the state estimate and covariance using the Kalman filter update step.

    Parameters:
    x_pred (np.ndarray): Predicted state estimate.
    P_pred (np.ndarray): Predicted estimate covariance.
    z (np.ndarray): Measurement vector.
    H (np.ndarray): Observation matrix.
    R (np.ndarray): Measurement noise covariance.

    Returns:
    tuple: Updated state, updated covariance, and Kalman gain.
    """
    
    y = z - H @ x_pred                      #* Innovation
    S = H @ P_pred @ H.T + R                #* Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)     #* Kalman gain

    x_updated = x_pred + K @ y
    P_updated = (np.eye(len(P_pred)) - K @ H) @ P_pred @ (np.eye(len(P_pred)) - K @ H).T + K @ R @ K.T #* Joseph formula | numerical stable

    return x_updated, P_updated, K