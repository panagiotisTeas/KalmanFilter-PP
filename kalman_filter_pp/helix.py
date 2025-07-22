from typing import Callable, Any

import numpy as np

from .logger import get_logger

logger = get_logger()

class Helix:
    """
    Represents a helix in a particle tracking system.
    This class encapsulates the state of a helix, including its parameters and arc length.
    The helix is defined by the parameters:
    - d0: Distance of closest approach (mm)
    - phi0: Initial azimuthal angle (rad)
    - omega: Helix curvature (mm^-1)
    - z0: Initial z position (mm)
    - cotTheta: Cotangent of the helix's polar angle

    Attributes
    ----------
    state (np.ndarray): Array of helix parameters [d0, phi0, omega, z0, cotTheta].

    arclength (float): Arc length from the perigee to the particle's initial position.

    Methods
    -------
    get_state() -> np.ndarray: Returns the current helix state parameters.

    get_arclength() -> float: Returns the current arc length.

    update_state(new_state: np.ndarray) -> None: Updates the helix state with new parameters.

    update_arclength(new_arclength: float) -> None: Updates the arc length value.

    __str__() -> str: Returns a string representation of the helix parameters.
    """

    def __init__(self, seeding : Callable[..., Any], *args: Any) -> None:
        """
        Initializes the Helix object with a seeding function and its arguments.

        Parameters
        ----------
        seeding (Callable[..., Any]) : Function to seed the helix parameters.

        *args (Any) : Arguments to be passed to the seeding function.
        
        Returns
        -------
        None
        
        Notes
        -----
        The seeding function should return a tuple containing the helix state parameters and the arc length.
        The state parameters should be in the order: [d0, phi0, omega, z0, cotTheta].
        The arc length is the distance from the perigee to the particle's initial position.
        
        Example usage
        -------
        >>> def seeding_function():        ...     return np.array([0.1, 0.5, 0.01, 0.0, 1.0]), 10.0
        >>> helix = Helix(seeding_function)
        >>> print(helix.get_state())
        [0.1 0.5 0.01 0.0 1.0]
        >>> print(helix.get_arclength())
        10.0
        """

        self.state, self.arclength = seeding(*args)

    def get_state(self) -> np.ndarray:
        """
        Returns the current helix state parameters.

        Returns
        -------
        np.ndarray: Array of helix parameters [d0, phi0, omega, z0, cotTheta].

        Example usage
        -------
        >>> helix = Helix(seeding_function)
        >>> state = helix.get_state()
        >>> print(state)
        [0.1 0.5 0.01 0.0 1.0]
        >>> print(state[0])  # d0
        0.1
        >>> print(state[1])  # phi0
        0.5
        >>> print(state[2])  # omega
        0.01
        >>> print(state[3])  # z0
        0.0
        >>> print(state[4])  # cotTheta
        1.0
        """

        return self.state
    
    def get_arclength(self) -> float:
        """
        Returns the current arc length from the perigee to the particle's initial position.

        Returns
        -------
        float: Arc length value.

        Example usage
        -------
        >>> helix = Helix(seeding_function)
        >>> arclength = helix.get_arclength()
        >>> print(arclength)
        10.0
        """

        return self.arclength
    
    def update_state(self, new_state : np.ndarray) -> None:
        """
        Updates the helix state with new parameters.

        Parameters
        ----------
        new_state (np.ndarray): Array of new helix parameters [d0, phi0, omega, z0, cotTheta].

        Returns
        -------
        None

        Example usage
        -------
        >>> helix = Helix(seeding_function)
        >>> new_state = np.array([0.2, 0.6, 0.02, 0.1, 1.5])
        >>> helix.update_state(new_state)
        >>> print(helix.get_state())
        [0.2 0.6 0.02 0.1 1.5]
        """

        self.state = new_state

    def update_arclength(self, z : float) -> None:
        """
        Updates the arc length value.

        Parameters
        ----------
        new_arclength (float): New arc length value.

        Returns
        -------
        None

        Example usage
        -------
        >>> helix = Helix(seeding_function)
        >>> helix.update_arclength(15.0)
        >>> print(helix.get_arclength())
        15.0
        """

        self.arclength = (z - self.state[3]) / self.state[4]

    def __str__(self) -> str:
        return f"Parameters: [d0 = {self.state[0]}(mm), phi0 = {self.state[1]}(rad), omega = {self.state[2]}(mm^-1), z0 = {self.state[3]}(mm), cotTheta = {self.state[4]}]"

#* The helix seeding is from Applied Fitting Theory IV | Formulas for Track Fitting by Paul Avery

def helix_seeding(position : np.ndarray, momentum : np.ndarray, charge : int, B_field : float, x_error_percent : float = 0, y_error_percent : float = 0, z_error_percent : float = 0) -> np.ndarray:
    """
    Seeds a helix from the given position and momentum of a particle. 
    The function applies Gaussian errors to the momentum components based on the specified error percentages.
    The helix parameters are calculated based on the particle's position, momentum, charge, and magnetic field strength.

    Parameters
    ----------
    position (np.ndarray): Particle's position in the form [x, y, z]

    momentum (np.ndarray): Particle's momentum in the form [px, py, pz]

    charge (int): Particle's charge (+/-1)

    B_field (float): Magnetic field strength in Tesla

    x_error_percent (float): Percentage error for the x component of momentum (default is 0)

    y_error_percent (float): Percentage error for the y component of momentum (default is 0)

    z_error_percent (float): Percentage error for the z component of momentum (default is 0)

    Returns
    -------
    np.ndarray: Helix parameters in the form [d0, phi0, omega, z0, cotTheta]

    float: Arc length from the perigee to the particle's initial position

    Example usage
    -------
    >>> position = np.array([1.0, 2.0, 3.0])  # Particle's position in mm
    >>> momentum = np.array([0.1, 0.2, 0.3])  # Particle's momentum in GeV
    >>> charge = 1  # Charge of the particle 
    >>> B_field = 2  # Magnetic field strength in Tesla
    >>> helix_params, arc_length = helix_seeding(position, momentum, charge, B_field, x_error_percent=1, y_error_percent=1, z_error_percent=1)
    """

    x, y, z = position
    px, py, pz = momentum
    
    #* Gaussian errors
    px = np.random.normal(px, x_error_percent * 1e-2 * np.abs(px))
    py = np.random.normal(py, y_error_percent * 1e-2 * np.abs(py))
    pz = np.random.normal(pz, z_error_percent * 1e-2 * np.abs(pz))
    
    q = charge

    a = -0.3 * B_field * q * 1e-3
    if a == 0:
        raise ValueError("Helix (a) parameter cannot be zero. Check the charge and magnetic field strength.")

    a_inv = 1 / a

    pt = np.hypot(px, py)
    if pt == 0:
        raise ValueError("Transverse momentum (pt) cannot be zero for helix seeding.")
    
    pt_inv = 1 / pt

    t = np.sqrt(pt**2 - 2 * a * (x * py - y * px) + (a**2) * (x**2 + y**2))

    d0 = a_inv * (t - pt)
    phi0 = np.atan2(py - a * x, px + a * y)
    omega = a * pt_inv
    cotTheta = pz * pt_inv
    s = (1 / omega) * np.atan2(omega * (x * px + y * py), pt - omega * (x * py - y * px))
    z0 = z - s * cotTheta

    return np.array([d0, phi0, omega, z0, cotTheta]), s

def parameters_transformation(parameters : np.ndarray, sign : int, mode : int = 0) -> np.ndarray:
    """
    Transforms the helix parameters between two representations:
    - From [d0, phi0, omega, z0, cotTheta] -> [d1, phi0, omega~, z0, cotTheta]
    - From [d1, phi0, omega~, z0, cotTheta] -> [d0, phi0, omega, z0, cotTheta]

    Parameters
    ----------
    parameters (np.ndarray): Array of helix parameters to be transformed.

    sign (int): Sign of the curvature (1 for positive, -1 for negative).
    
    mode (int): Transformation mode (0 for forward, 1 for backward). Default is 0.

    Returns
    -------
    np.ndarray: Transformed helix parameters in the specified mode.

    Example usage
    -------
    >>> params = np.array([0.1, 0.5, 0.01, 0.0, 1.0])  # [d0, phi0, omega, z0, cotTheta]
    >>> transformed_params = parameters_transformation(params, sign=1, mode=0)
    >>> print(transformed_params)  # [d1, phi0, omega~, z0, cotTheta]
    >>> transformed_params = parameters_transformation(transformed_params, sign=1, mode=1)
    >>> print(transformed_params)  # [d0, phi0, omega, z0, cotTheta]
    """

    if mode == 0:
        d0 = parameters[0]
        phi0 = parameters[1]
        omega = parameters[2]
        z0 = parameters[3]
        cot_theta = parameters[4]

        d1 = d0 * (1 + omega * d0 / 2) / (1 + omega * d0)
        omega_tilde = omega / (2 + 2 * omega * d0)

        return np.array([d1, phi0, omega_tilde, z0, cot_theta])

    else:
        d1 = parameters[0]
        phi0 = parameters[1]
        omega_tilde = parameters[2]
        z0 = parameters[3]
        cot_theta = parameters[4]

        omega = 2 * omega_tilde / np.sqrt(1 - 4 * omega_tilde * d1)
        d0 = 0
        if np.sign(omega) == sign:
            d0 = (1 - np.sqrt(1 - 4 * omega_tilde * d1)) / (2 * omega_tilde)
        else:
            omega = -omega
            d0 = (1 + np.sqrt(1 - 4 * omega_tilde * d1)) / (2 * omega_tilde)

        return np.array([d0, phi0, omega, z0, cot_theta])