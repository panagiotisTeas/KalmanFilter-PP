import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from tqdm import tqdm

from .logger import get_logger
from .helix import Helix
from .config import DETECTOR_IMPORT_PATH, PARTICLE_IMPORT_PATH

logger = get_logger()

def detector_xy_projection(ax : plt.Axes, volume_color : dict | None = None) -> None:
    """
    Plots the XY projection of detector volumes on the given axes. 
    Each volume is represented by a colored polygon based on its ID.

    Parameters
    -----------
    ax (plt.Axes): The matplotlib axes on which to plot the detector volumes.

    volume_color (dict, optional): A dictionary mapping volume IDs to colors. If None, a default color mapping is used.

    Returns
    -----------
    None

    Notes
    -----------
    The function reads volume data from a CSV file located at `DETECTOR_IMPORT_PATH + "cartesian.csv"`.
    The CSV file is expected to contain columns for volume IDs and their corner coordinates.

    Example usage
    -----------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_xy_projection(ax)
    >>> ax.axis("auto")
    >>> plt.show()
    """
    
    data = pd.read_csv(DETECTOR_IMPORT_PATH + "cartesian.csv")

    if volume_color is None:
        volume_color = {
            7 : "orange",
            8 : "lime",
            9 : "cornflowerblue",
            12 : "red", 
            13 : "green", 
            14 : "royalblue",
            16 : "maroon", 
            17 : "darkgreen", 
            18 : "blue",
        }
    
    for volume, color in volume_color.items():
        volume_data = data[data["volume_id"] == volume]

        if volume in [8, 13, 17]:
            ftl = volume_data[["ftl_x", "ftl_y"]].to_numpy()
            ftr = volume_data[["ftr_x", "ftr_y"]].to_numpy()
            btr = volume_data[["btr_x", "btr_y"]].to_numpy()
            btl = volume_data[["btl_x", "btl_y"]].to_numpy()
            corners = np.stack([ftl, ftr, btr, btl, ftl], axis=1)

        else:
            ftl = volume_data[["ftl_x", "ftl_y"]].to_numpy()
            ftr = volume_data[["ftr_x", "ftr_y"]].to_numpy()
            fbr = volume_data[["fbr_x", "fbr_y"]].to_numpy()
            fbl = volume_data[["fbl_x", "fbl_y"]].to_numpy()
            corners = np.stack([ftl, ftr, fbr, fbl, ftl], axis=1)

        segments = np.concatenate([
            np.stack([corners[:, i], corners[:, i + 1]], axis=1)
            for i in range(4)
        ], axis=0)

        lines = LineCollection(segments, colors=color, linewidths=0.5, alpha=0.5)
        ax.add_collection(lines)

def detector_rz_projection(ax : plt.Axes, vol_col : dict | None = None) -> None:
    """
    Plots the RZ projection of detector volumes on the given axes.
    Each volume is represented by a colored polygon based on its ID.

    Parameters
    -----------
    ax (plt.Axes): The matplotlib axes on which to plot the detector volumes.

    vol_col (dict, optional): A dictionary mapping volume IDs to colors. If None, a default color mapping is used.

    Returns
    -----------
    None

    Notes
    -----------
    The function reads volume data from a CSV file located at `DETECTOR_IMPORT_PATH + "cylindrical.csv"`.
    The CSV file is expected to contain columns for volume IDs and their corner coordinates.

    Example usage
    -----------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_rz_projection(ax)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    data = pd.read_csv(DETECTOR_IMPORT_PATH + "cylindrical.csv")

    if vol_col is None:
        volume_colors = {7 : "orange", 8 : "lime", 9 : "cornflowerblue",
                        12 : "red", 13 : "green", 14 : "royalblue",
                        16 : "maroon", 17 : "darkgreen", 18 : "blue",}
    else:
        volume_colors = vol_col

    for volume_id, color in volume_colors.items():
        volume_data = data[data["volume_id"] == volume_id]

        if volume_id in [8, 13, 17]:
            ftl = volume_data[["ftl_z", "ftl_r"]].to_numpy()
            ftr = volume_data[["ftr_z", "ftr_r"]].to_numpy()
            btr = volume_data[["fbr_z", "fbr_r"]].to_numpy()
            btl = volume_data[["fbl_z", "fbl_r"]].to_numpy()
            corners = np.stack([ftl, ftr, btr, btl, ftl], axis=1)
        else:
            ftl = volume_data[["ftl_z", "ftl_r"]].to_numpy()
            ftr = volume_data[["btl_z", "btl_r"]].to_numpy()
            fbr = volume_data[["bbl_z", "bbl_r"]].to_numpy()
            fbl = volume_data[["fbl_z", "fbl_r"]].to_numpy()
            corners = np.stack([ftl, ftr, fbr, fbl, ftl], axis=1)


        segments = np.concatenate([
            np.stack([corners[:, i], corners[:, i + 1]], axis=1)
            for i in range(4)
        ], axis=0)

        lines = LineCollection(segments, colors=color, linewidths=0.5, alpha=0.5)
        ax.add_collection(lines)

def particle_xy_projection_byId(ax : plt.Axes, path_in : str, particle_ids : list, rand_color : bool = True) -> None:
    """
    Adds the XY projection of particle hits to the given matplotlib axis.

    Parameters
    ----------
    ax (plt.Axes): Matplotlib axis object where the hits will be plotted.

    path_in (str): Filename of the particle data.

    particle_ids (list): List of particle IDs to render.

    rand_color (bool, optional): If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.

    Returns
    -------
    None

    Notes
    -----
    This function does not correctly handle particles that originate in the endcap region and later enter the barrel region — the hit ordering may appear visually misleading in such cases.

    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_xy_projection(ax)
    >>> particle_xy_projection_byId(ax, "event000001000.csv", [4503668346847232, 4503737066323968, 4505317614288896], rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    data = pd.read_csv(PARTICLE_IMPORT_PATH + path_in)
    data = data[data["particle_id"].isin(particle_ids)]
    data = data[data["hit_id"] != -1]

    volume_order = {vol_id : i for i, vol_id in enumerate([8, 13, 17, 7, 9, 12, 14, 16, 18])}

    for particle in particle_ids:
        particle_data : pd.DataFrame = data[data["particle_id"] == particle].copy()

        particle_data["r"] = np.sqrt(particle_data["x"]**2 + particle_data["y"]**2)
        particle_data["volume_order"] = particle_data["volume_id"].map(volume_order)
        particle_data["sort_key"] = np.where(particle_data["volume_id"].isin([8, 13, 17]), particle_data["r"], particle_data["z"])
        particle_data = particle_data.sort_values(by=["volume_order", "sort_key"], ignore_index=True)

        if rand_color is True:
            color = np.array(np.random.choice(range(256), size=3)) / 256
        else:
            color = np.array([1.0, 0.0, 0.0])

        hit = particle_data[["x", "y"]].to_numpy()

        segments = np.stack([hit[:-1], hit[1:]], axis=1)

        lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
        ax.add_collection(lines)

def particle_xy_projection_byEvents(ax : plt.Axes, path_in : str, momentum_cutoff : float = 2, rand_color : bool = True) -> None:
    """
    Adds the XY projection of particle hits to the given matplotlib axis.

    Parameters
    ----------
    ax (plt.Axes): Matplotlib axis object where the hits will be plotted.

    path_in (str): Filename of the particle data.

    momentum_cutoff (float): Excludes all particles with momentum less than the specified value (default 2 GeV).

    rand_color (bool): If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.

    Returns
    -------
    None

    Notes
    -----
    This function does not correctly handle particles that originate in the endcap region and later enter the barrel region — the hit ordering may appear visually misleading in such cases.

    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_xy_projection(ax)
    >>> particle_xy_projection_byEvents(ax, "event000001000.csv", momentum_cutoff=10, rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    data = pd.read_csv(PARTICLE_IMPORT_PATH + path_in)
    data = data[data["hit_id"] != -1]

    data["momentum"] = np.sqrt(data["tpx"]**2 + data["tpy"]**2 + data["tpz"]**2)
    data = data[data["momentum"] >= momentum_cutoff]

    particle_ids = set(data["particle_id"])

    volume_order = {vol_id : i for i, vol_id in enumerate([8, 13, 17, 7, 9, 12, 14, 16, 18])}

    for particle in tqdm(particle_ids):
        particle_data : pd.DataFrame = data[data["particle_id"] == particle].copy()

        particle_data["r"] = np.sqrt(particle_data["x"]**2 + particle_data["y"]**2)
        particle_data["volume_order"] = particle_data["volume_id"].map(volume_order)
        particle_data["sort_key"] = np.where(particle_data["volume_id"].isin([8, 13, 17]), particle_data["r"], particle_data["z"])
        particle_data = particle_data.sort_values(by=["volume_order", "sort_key"], ignore_index=True)

        if rand_color is True:
            color = np.array(np.random.choice(range(256), size=3)) / 256
        else:
            color = np.array([1.0, 0.0, 0.0])

        hit = particle_data[["x", "y"]].to_numpy()

        segments = np.stack([hit[:-1], hit[1:]], axis=1)

        lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
        ax.add_collection(lines)

def particle_rz_projection_byId(ax : plt.Axes, path_in : str, particle_ids : list, rand_color : bool = True) -> None:
    """
    Adds the RZ projection of particle hits to the given matplotlib axis.
    
    Parameters
    ----------
    ax (plt.Axes) : Matplotlib axis object where the hits will be plotted.

    path_in (str) : Filename of the particle data.

    particle_ids (list) : List of particle IDs to render.

    rand_color (bool) : If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.
    
    Returns
    -------
    None
    
    Notes
    -----
    This function does not correctly handle particles that originate in the endcap region and later enter the barrel region — the hit ordering may appear visually misleading in such cases.
    
    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_rz_projection(ax)
    >>> particle_rz_projection_byId(ax, "event000001000.csv", [4503668346847232, 4503737066323968, 4505317614288896], rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    data = pd.read_csv(PARTICLE_IMPORT_PATH + path_in)
    data = data[data["particle_id"].isin(particle_ids)]
    data = data[data["hit_id"] != -1]

    volume_order = {vol_id : i for i, vol_id in enumerate([8, 13, 17, 7, 9, 12, 14, 16, 18])}

    for particle in particle_ids:
        particle_data : pd.DataFrame = data[data["particle_id"] == particle].copy()

        particle_data["r"] = np.sqrt(particle_data["x"]**2 + particle_data["y"]**2)
        particle_data["volume_order"] = particle_data["volume_id"].map(volume_order)
        particle_data["sort_key"] = np.where(particle_data["volume_id"].isin([8, 13, 17]), particle_data["r"], particle_data["z"])
        particle_data = particle_data.sort_values(by=["volume_order", "sort_key"], ignore_index=True)

        if rand_color is True:
            color = np.array(np.random.choice(range(256), size=3)) / 256
        else:
            color = np.array([1.0, 0.0, 0.0])

        hit = particle_data[["z", "r"]].to_numpy()

        segments = np.stack([hit[:-1], hit[1:]], axis=1)

        lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
        ax.add_collection(lines)

def particle_rz_projection_byEvents(ax : plt.Axes, path_in : str, momentum_cutoff : float = 2, rand_color : bool = True) -> None:
    """
    Adds the RZ projection of particle hits to the given matplotlib axis.

    Parameters
    ----------
    ax (plt.Axes): Matplotlib axis object where the hits will be plotted.

    path_in (str): Filename of the particle data.

    momentum_cutoff (float): Excludes all particles with momentum less than the specified value (default 2 GeV).

    rand_color (bool): If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.

    Returns
    -------
    None

    Notes
    -----
    This function does not correctly handle particles that originate in the endcap region and later enter the barrel region — the hit ordering may appear visually misleading in such cases.

    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> detector_rz_projection(ax)
    >>> particle_rz_projection_byEvents(ax, "event000001000.csv", momentum_cutoff=10, rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    data = pd.read_csv(PARTICLE_IMPORT_PATH + path_in)
    data = data[data["hit_id"] != -1]

    data["momentum"] = np.sqrt(data["tpx"]**2 + data["tpy"]**2 + data["tpz"]**2)
    data = data[data["momentum"] >= momentum_cutoff]

    particle_ids = set(data["particle_id"])

    volume_order = {vol_id : i for i, vol_id in enumerate([8, 13, 17, 7, 9, 12, 14, 16, 18])}

    for particle in tqdm(particle_ids):
        particle_data : pd.DataFrame = data[data["particle_id"] == particle].copy()

        particle_data["r"] = np.sqrt(particle_data["x"]**2 + particle_data["y"]**2)
        particle_data["volume_order"] = particle_data["volume_id"].map(volume_order)
        particle_data["sort_key"] = np.where(particle_data["volume_id"].isin([8, 13, 17]), particle_data["r"], particle_data["z"])
        particle_data = particle_data.sort_values(by=["volume_order", "sort_key"], ignore_index=True)

        if rand_color is True:
            color = np.array(np.random.choice(range(256), size=3)) / 256
        else:
            color = np.array([1.0, 0.0, 0.0])

        hit = particle_data[["z", "r"]].to_numpy()

        segments = np.stack([hit[:-1], hit[1:]], axis=1)

        lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
        ax.add_collection(lines)

def helix_xy_trajectory(ax : plt.Axes, helix : Helix, rand_color : bool = True) -> None:
    """
    Adds the helix trajectory projection to the given matplotlib axis.
    
    Parameters
    ----------
    ax (plt.Axes): Matplotlib axis object where the hits will be plotted.

    helix (Helix): Helix object containing the state parameters and arc length of the particle.

    rand_color (bool, optional): If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.

    Returns
    -------
    None

    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> helix = Helix(seeding_function)
    >>> helix_xy_trajectory(ax, helix, rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    state = helix.get_state()
    arclength = helix.get_arclength()

    arclength_segments = np.linspace(arclength, arclength + 1000, 20) #* [arclength -> arclength + 2000] (mm)

    #* parametric equations of helix
    x = - state[0] * np.sin(state[1]) + (1 / state[2]) * (np.sin(state[2] * arclength_segments + state[1]) - np.sin(state[1]))
    y = + state[0] * np.cos(state[1]) - (1 / state[2]) * (np.cos(state[2] * arclength_segments + state[1]) - np.cos(state[1]))
    # z = state[3] + state[4] * arclength_segments #* not needed

    hit = np.stack((x, y), axis=1)
    segments = np.stack([hit[:-1], hit[1:]], axis=1)


    if rand_color is True:
        color = np.array(np.random.choice(range(256), size=3)) / 256
    else:
        color = np.array([1.0, 0.0, 0.0])

    lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
    ax.add_collection(lines)

def helix_rz_trajectory(ax : plt.Axes, helix : Helix, rand_color : bool = True) -> None:
    """
    Adds the helix trajectory projection to the given matplotlib axis.
    
    Parameters
    ----------
    ax (plt.Axes): Matplotlib axis object where the hits will be plotted.

    helix (Helix): Helix object containing the state parameters and arc length of the particle.

    rand_color (bool, optional): If True, assigns a random RGB color to each particle. Otherwise, uses red as the default color.

    Returns
    -------
    None

    Example usage
    --------
    >>> fig, ax = plt.subplots(figsize=(10, 8))
    >>> helix = Helix(seeding_function)
    >>> helix_rz_trajectory(ax, helix, rand_color=True)
    >>> ax.axis("auto")
    >>> plt.show()
    """

    state = helix.get_state()
    arclength = helix.get_arclength()

    arclength_segments = np.linspace(arclength, arclength + 1000, 20) #* [arclength -> arclength + 2000] (mm)

    #* parametric equations of helix
    x = - state[0] * np.sin(state[1]) + (1 / state[2]) * (np.sin(state[2] * arclength_segments + state[1]) - np.sin(state[1]))
    y = + state[0] * np.cos(state[1]) - (1 / state[2]) * (np.cos(state[2] * arclength_segments + state[1]) - np.cos(state[1]))
    z = state[3] + state[4] * arclength_segments

    r = np.sqrt(x**2 + y**2)

    hit = np.stack((z, r), axis=1)
    segments = np.stack([hit[:-1], hit[1:]], axis=1)


    if rand_color is True:
        color = np.array(np.random.choice(range(256), size=3)) / 256
    else:
        color = np.array([1.0, 0.0, 0.0])

    lines = LineCollection(segments, colors=color, linewidths=0.8, alpha=1)
    ax.add_collection(lines)