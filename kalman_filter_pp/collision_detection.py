import numpy as np
import pandas as pd

from rtree import index
from scipy.spatial import Delaunay

from .logger import get_logger
from .config import BOUNDS_IMPORT_PATH, DETECTOR_IMPORT_PATH
from .helix import Helix

logger = get_logger()

def build_rtree() -> index.Index:
    """
    Builds an R-tree index from the bounds data.
    The bounds data is expected to be in a CSV file located at the path specified by BOUNDS_IMPORT_PATH.
    The CSV should contain columns: 'rmin', 'rmax', 'zmin', 'zmax', 'volume_id', 'layer_id', and 'module_id'.
    Each row in the CSV represents a detector volume with its spatial bounds and identifiers.

    Returns
    -------
    index.Index: An R-tree index containing the detector volumes.

    Example usage
    -------
    >>> rtree_idx = build_rtree()
    """

    bounds = pd.read_csv(BOUNDS_IMPORT_PATH + "bounds.csv")

    p = index.Property()
    p.dimension = 2
    rtree_idx = index.Index(properties=p)

    for idx, row in bounds.iterrows():
        rmin, rmax = row['rmin'], row['rmax']
        zmin, zmax = row['zmin'], row['zmax']

        detector_id = (int(row["volume_id"]), int(row["layer_id"]), int(row["module_id"]))
        rtree_idx.insert(0, (rmin, zmin, rmax, zmax), obj=detector_id)

    return rtree_idx

def collision_detection_2d(r: float, z: float, rtree_idx: index.Index) -> list:
    """
    Detects collisions in 2D space using an R-tree index.

    Parameters
    ----------
    r (float): The radial coordinate of the point to check for collisions.

    z (float): The axial coordinate of the point to check for collisions.

    rtree_idx (index.Index): The R-tree index containing the detector volumes.

    Returns
    -------
    list: A list of tuples containing the detector IDs that collide with the point (r, z).

    Example usage
    -------
    >>> rtree_idx = build_rtree()
    >>> r, z = 10.0, 5.0
    >>> collisions = collision_detection_2d(r, z, rtree_idx)
    >>> print(collisions)
    """
    
    results = list(rtree_idx.intersection((r, z, r, z), objects=True))

    return [item.object for item in results]

def build_delaunay() -> dict:
    """
    Builds a Delaunay triangulation for each detector volume based on its corner points.
    The detector data is expected to be in a CSV file located at the path specified by DETECTOR_IMPORT_PATH.
    The CSV should contain columns for the front and back corners of each detector volume.
    Each row in the CSV represents a detector volume with its corner points and identifiers.
    
    Returns
    -------
    dict: A dictionary where keys are tuples of (volume_id, layer_id, module_id)
          and values are Delaunay triangulation objects for the corresponding detector volume.
          
    Example usage
    -------
    >>> delaunay_dict = build_delaunay()
    """

    detector = pd.read_csv(DETECTOR_IMPORT_PATH + "cartesian.csv")
    delaunay_dict = {}

    for _, row in detector.iterrows():
        points = np.array([
            [row["fbl_x"], row["fbl_y"], row["fbl_z"]], [row["fbr_x"], row["fbr_y"], row["fbr_z"]],
            [row["ftl_x"], row["ftl_y"], row["ftl_z"]], [row["ftr_x"], row["ftr_y"], row["ftr_z"]],
            [row["bbl_x"], row["bbl_y"], row["bbl_z"]], [row["bbr_x"], row["bbr_y"], row["bbr_z"]],
            [row["btl_x"], row["btl_y"], row["btl_z"]], [row["btr_x"], row["btr_y"], row["btr_z"]]
        ])
        key = (row["volume_id"], row["layer_id"], row["module_id"])
        delaunay_dict[key] = Delaunay(points)

    return delaunay_dict

def collision_detection_3d(x : float, y : float, z : float, candidate_keys : list, delaunay_dict : dict) -> tuple | None:

    for key in candidate_keys:
        tri = delaunay_dict.get(key)
        if tri and tri.find_simplex(np.array([[x, y, z]])) >= 0:
            return key
    return None

def collision_detection(helix : Helix, rtree_idx: index.Index, delaunay : dict, current : np.ndarray) -> tuple:
    """
    Detects collisions along the helix trajectory using the R-tree index and Delaunay triangulation.

    Parameters
    ----------
    helix (Helix): The Helix object containing the trajectory parameters.

    rtree_idx (index.Index): The R-tree index containing the detector volumes.

    delaunay (dict): A dictionary of Delaunay triangulations for each detector volume.

    current (np.ndarray): An array containing the current detector volume and layer IDs.

    Returns
    -------
    tuple: A tuple containing the hit position (np.ndarray) and the detector IDs (np.ndarray).
           If no collision is detected, returns (None, None).

    Example usage
    -------
    >>> helix = Helix(seeding_function)
    >>> rtree_idx = build_rtree()
    >>> delaunay = build_delaunay()
    >>> current = np.array([8, 2])  # Current detector volume and layer
    >>> hit, detector_ids = collision_detection(helix, rtree_idx, delaunay, current)
    """
    
    parameters = helix.get_state()
    d0, phi0, omega, z0, cotTheta = parameters
    arclength_segments = np.linspace(0, 1200, 18000) #! The numbers are hardcoded because these make sure that we do not miss any collisions

    x = - d0 * np.sin(phi0) + (1 / omega) * (np.sin(phi0 + omega * arclength_segments) - np.sin(phi0))
    y =   d0 * np.cos(phi0) - (1 / omega) * (np.cos(phi0 + omega * arclength_segments) - np.cos(phi0))
    z = z0 + arclength_segments * cotTheta
    r = np.sqrt(x**2 + y**2)

    current_volume = current[0]
    current_layer = current[1]

    target_volume = None
    target_layer = None

    passed_current = False
    hits = []

    #* Iterate through the points along the helix trajectory
    for rp, xp, yp, zp in zip(r, x, y, z):
        
        #* Check if the point is within the bounds of the current detector
        possible_collisions = collision_detection_2d(rp, zp, rtree_idx)
        collision = collision_detection_3d(xp, yp, zp, possible_collisions, delaunay)

        #* If no collision is detected, continue to the next point
        if collision is None:
            continue

        # logger.debug(f"Collision detected at (x={xp}, y={yp}, z={zp}) with detector {collision}")

        volume_id, layer_id, _ = collision

        #* Check if we have passed the current detector
        if passed_current is False:
            if (volume_id, layer_id) == (current_volume, current_layer):
                passed_current = True

            continue

        #* If we have passed the current detector, check if we are still in the same detector
        if target_volume is None or target_layer is None:
            #* If we are in a different detector, store the hit and continue
            if (volume_id, layer_id) != (current_volume, current_layer):
                target_volume = volume_id
                target_layer = layer_id
                hits.append([xp, yp, zp])
        else:
            #* If we are still in the same detector, continue collecting hits
            if (volume_id, layer_id) == (target_volume, target_layer):
                hits.append([xp, yp, zp])
            else:
                break
    
    #* If no hits were collected, return None
    if not hits:
        logger.debug("No collisions detected after the current detector.")
        return None, None
    
    hits = np.array(hits, dtype=np.float32)
    hit = np.mean(hits, axis=0)
    detector_ids = np.array([target_volume, target_layer], dtype=np.int32)

    return hit, detector_ids

