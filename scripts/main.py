import numpy as np
import pandas as pd

from kalman_filter_pp.config import PARTICLE_IMPORT_PATH
from kalman_filter_pp.collision_detection import build_rtree, build_delaunay, collision_detection
from kalman_filter_pp.helix import Helix, helix_seeding

df = pd.read_csv(PARTICLE_IMPORT_PATH + "event000001000.csv")

rtree = build_rtree()
delaunay = build_delaunay()

for index, row in df.iterrows():
    hit = row["hit_id"]
    
    if hit != -1:
        continue

    position = np.array([row["x"], row["y"], row["z"]])
    momentum = np.array([row["tpx"], row["tpy"], row["tpz"]])
    charge = row["q"]
    magnetic_field = 2

    helix = Helix(helix_seeding, position, momentum, charge, magnetic_field)

    hit, det_id = collision_detection(helix, rtree, delaunay, np.array([8, 2]))

    print(f"Hit: {hit}, Detector ID: {det_id}")

    break

