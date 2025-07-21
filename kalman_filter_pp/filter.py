import numpy as np
import pandas as pd

from .errors import ErrorFlags

def make_spacepoints(data : np.ndarray) -> np.ndarray:
    """
    Create spacepoints from the given data.
    
    Parameters
    ----------
    data (np.ndarray): Input data containing hit information.
    
    Returns
    -------
    np.ndarray: Processed spacepoints with averaged values.
    
    Example usage
    ----------
    >>> data = np.array([[8, 2, 1.0, 2.0, 3.0],
    ...                  [8, 2, 1.5, 2.5, 3.5],
    ...                  [8, 3, 1.0, 2.0, 3.0],
    ...                  [8, 3, 1.5, 2.5, 3.5]])
    >>> make_spacepoints(data)
    [[8. , 2. , 1.25, 2.25, 3.25],
     [8. , 3. , 1.25, 2.25, 3.25]]
    """
    
    df = pd.DataFrame(data)
    grouped = df.groupby([0, 1], as_index=False).mean()
    return grouped.to_numpy()

#! 5 because we have the vertex and require at least 4 hits
def filter_spacepoints(data : np.ndarray, flags : ErrorFlags, amount : int = 5) -> None:
    """
    Filter spacepoints based on the number of hits.

    Parameters
    ----------
    data (np.ndarray): Input data containing spacepoints.

    flags (ErrorFlags): Error flags to set if the condition is met.

    amount (int): Minimum number of hits required to not set the flag.

    Returns
    -------
    None: The function modifies the flags in place.

    Example usage
    ----------
    >>> data = np.array([[8, 2, 1.0, 2.0, 3.0],
    ...                  [8, 4, 1.5, 2.5, 3.5],
    ...                  [8, 6, 1.0, 2.0, 3.0],
    ...                  [8, 8, 1.5, 2.5, 3.5]])
    >>> flags = ErrorFlags(0b00000000)
    >>> filter_spacepoints(data, flags, amount=4)
    >>> print(flags.bit_0)  # Should be 0 if there are at least 4 hits
    """

    if data.shape[0] < amount:
        flags.bit_0 = 1

def filter_momentum(data : np.ndarray, flags : ErrorFlags, momentum : float = 2) -> None:
    """
    Filter spacepoints based on the momentum of the particle.

    Parameters
    ----------
    data (np.ndarray): Input data containing particle information.

    flags (ErrorFlags): Error flags to set if the condition is met.

    momentum (float): Minimum momentum threshold.

    Returns
    -------
    None: The function modifies the flags in place.

    Example usage
    ----------
    >>> data = np.array([[1.0, 2.0, 3.0],
    ...                  [1.5, 2.5, 3.5],
    ...                  [1.0, 2.0, 3.0],
    ...                  [1.5, 2.5, 3.5]])
    >>> flags = ErrorFlags(0b00000000)
    >>> filter_momentum(data, flags, momentum=2)
    >>> print(flags.bit_1)  # Should be 0 if the momentum is above the threshold
    """

    if np.linalg.norm(data) < momentum:
        flags.bit_1 = 1

def filter_region(data : np.ndarray, flags : ErrorFlags, lower_theta : float = 75, upper_theta : float = 90, lower_phi : float = 0, upper_phi : float = 20) -> None:
    """
    Filter spacepoints based on the region defined by theta and phi bounds.

    Parameters
    ----------
    data (np.ndarray): Input data containing spacepoints.

    flags (ErrorFlags): Error flags to set if the condition is met.

    lower_theta (float): Lower bound for theta.

    upper_theta (float): Upper bound for theta.

    lower_phi (float): Lower bound for phi.

    upper_phi (float): Upper bound for phi.

    Returns
    -------
    None: The function modifies the flags in place.

    Example usage
    ----------
    >>> data = np.array([[8.0, 2.0, 1.0, 2.0, 3.0],
    ...                  [8.0, 4.0, 1.5, 2.5, 3.5],
    ...                  [8.0, 6.0, 1.0, 2.0, 3.0],
    ...                  [8.0, 8.0, 1.5, 2.5, 3.5]])
    >>> flags = ErrorFlags(0b00000000)
    >>> filter_region(data, flags, lower_theta=75, upper_theta=90, lower_phi=0, upper_phi=20)
    >>> print(flags.bit_2)  # Should be 0 if the spacepoints are within the bounds
    """

    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]
    r = np.sqrt(x**2 + y**2 + z**2)

    theta = np.degrees(np.arccos(z / r))
    phi = np.degrees(np.atan2(y, x))

    theta_in_bounds = np.all((theta > lower_theta) & (theta < upper_theta))
    phi_in_bounds = np.all((phi > lower_phi) & (phi < upper_phi))

    if not (theta_in_bounds and phi_in_bounds):
        flags.bit_2 = 1