import os
import csv

import numpy as np
import pandas as pd

from tqdm import tqdm

from .errors import ErrorFlags
from .logger import get_logger
from .config import KALMAN_FILTER_EXPORT_PATH, PARTICLE_IMPORT_PATH
from .filter import make_spacepoints, filter_spacepoints, filter_momentum, filter_region, choose_spacepoint
from .helix import Helix, helix_seeding, parameters_transformation
from .matrices import F, P, Q, R, get_H
from .jacobians import forward_jacobian, backward_jacobian, qpt_jacobian, omega_jacobian
from .collision_detection import build_rtree, build_delaunay, collision_detection
from .hit_miss import hit_miss
from .linear_kalman import kf_predict, kf_update

logger = get_logger()

def linear_kalman_algorithm(event_start : int = 1000, event_end : int = 1050, spec_event : int = None, spec_particle : int = None, magnetic_field : float = 2, particle_iterations : int = 10000000, output : str = KALMAN_FILTER_EXPORT_PATH, lower_theta : float = 70, lower_phi : float = 0, upper_theta : float = 110, upper_phi : float = 180) -> None:
    
    #* Check if the output file exists, if not create it with headers
    if not os.path.exists(output):
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["event_number", "particle_id", "is_vertex"
                            #* Track parameters
                            "d0", "phi0", "q/pT", "z0", "theta",
                            #* Covariance matrix P
                            "P00", "P01", "P02", "P03", "P04",
                            "P10", "P11", "P12", "P13", "P14",
                            "P20", "P21", "P22", "P23", "P24", 
                            "P30", "P31", "P32", "P33", "P34",
                            "P40", "P41", "P42", "P43", "P44", 
                            #* Kalman gain K
                            "K00", "K01",
                            "K10", "K11",
                            "K20", "K21", 
                            "K30", "K31", 
                            "K40", "K41"])
    
    #* Initialize the iteration counter
    iteration = 1

    #* Initialize the R-tree and Delaunay triangulation
    rtree = build_rtree()
    delaunay = build_delaunay()

    #* If a specific event is specified, look only at that event
    if spec_event is not None:
        event_start = spec_event
        event_end = spec_event + 1

    #* Iterate over the events
    for event_number in tqdm(range(event_start, event_end)):

        #* Load the particle data for the event
        event_data = pd.read_csv(PARTICLE_IMPORT_PATH + f"event00000{event_number}.csv")

        vertex_data = event_data[event_data['hit_id'] == -1].reset_index(drop=True)

        #! -------------------------------------------------------------------

        x = event_data['x'].to_numpy()
        y = event_data['y'].to_numpy()
        z = event_data['z'].to_numpy()

        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.degrees(np.arccos(z / r))
        phi = np.degrees(np.arctan2(y, x)) % 360

        theta_mask = (theta > 70) & (theta < 110)
        phi_mask = (phi > 0) & (phi < 180)
        full_mask = theta_mask & phi_mask

        event_data = event_data[full_mask].reset_index(drop=True)

        #! -------------------------------------------------------------------

        p = np.linalg.norm(event_data[['tpx', 'tpy', 'tpz']].values, axis=1)
        mask = p >= 2
        event_data = event_data[mask].reset_index(drop=True)

        #! -------------------------------------------------------------------

        #* Get unique particle IDs in the event
        event_particle_ids = np.unique(event_data["particle_id"].to_numpy())
        logger.info(f"Event {event_number} has {len(event_particle_ids)} particles.")

        #* If a specific particle ID is specified, filter the particle IDs
        if spec_particle is not None:
            event_particle_ids = event_particle_ids[event_particle_ids == spec_event]

        #* Iterate over the particle IDs
        for particle_id in tqdm(event_particle_ids):
            
            if iteration > particle_iterations:
                logger.info(f"Reached particle iteration limit: {particle_iterations}")
                return

            errors = ErrorFlags(0b00000000)
            rows_to_write = []

            #* Get the particle data for the current particle ID
            particle_data = event_data[event_data["particle_id"] == particle_id].to_numpy()

            #* Get the measurements for the particle and create spacepoints
            hit_measurements = particle_data[:, [2,3,5,6,7]]  #* 2: volume_id, 3: layer_id, 5: x, 6: y, 7: z
            spacepoint_measurements = make_spacepoints(hit_measurements)

            #* Filtering spacepoints
            filter_spacepoints(spacepoint_measurements, errors, amount=5)
            if errors.bit_0:
                logger.debug(f"Event {event_number}, particle {particle_id} has low spacepoints: {len(spacepoint_measurements)}")
                continue

            vertex = vertex_data[vertex_data['particle_id'] == particle_id].to_numpy().flatten()

            seeding_momentum = vertex[11:14]  #* 11: px, 12: py, 13: pz
            seeding_charge = vertex[14]       #* 14: charge

            #* Filtering momentum
            # filter_momentum(seeding_momentum, errors, momentum=2)
            # if errors.bit_1:
            #     logger.debug(f"Event {event_number}, particle {particle_id} has low momentum: {np.linalg.norm(seeding_momentum)}")
            #     continue

            #* Filtering region
            # filter_region(spacepoint_measurements, errors, 70, 110, 0, 180)
            # if errors.bit_2:
            #     continue

            #* Write the truth parameters as the first row to every particle
            helix_truth = Helix(helix_seeding, vertex[2:5], seeding_momentum, seeding_charge, magnetic_field)
            rows_to_write.append([event_number, particle_id, 1] + 
                                helix_truth.get_state().flatten().tolist() +
                                P.flatten().tolist() +
                                [0.0] * (5 * 2))  #* Placeholder for Kalman gain K
            
            #* Initialize the Kalman filter with the first spacepoint
            helix = Helix(helix_seeding, spacepoint_measurements[0, 2:5], seeding_momentum, seeding_charge, magnetic_field)
            position_at_detector = spacepoint_measurements[0, :2]
            spacepoint_measurements = spacepoint_measurements[1:]

            rows_to_write.append([event_number, particle_id, 0] + 
                                helix.get_state().flatten().tolist() +
                                P.flatten().tolist() +
                                [0.0] * (5 * 2))  #* Placeholder for Kalman gain K
            
            omega_sign = np.sign(helix.get_state()[2])

            #* Initialize the covariance matrix
            P_cov = P
            
            #* Iterate over the spacepoints
            #TODO: Increase the iteration counter
            while spacepoint_measurements.shape[0] > 0:

                logger.debug(f"Event {event_number}, particle {particle_id}")
                
                #* Transform the state and covariance matrix from [d0, phi0, q/pt, z0, theta] -> [d1, phi0, omega~, z0, cotTheta]
                P_transformed = forward_jacobian(helix) @ qpt_jacobian(helix) @ P_cov @ qpt_jacobian(helix).T @ forward_jacobian(helix).T
                state_transformed = parameters_transformation(helix.get_state(), omega_sign, 0)
                #* Kalman filter prediction step
                state_predicted, P_predicted = kf_predict(state_transformed, P_transformed, F, forward_jacobian(helix) @ qpt_jacobian(helix) @ Q @ qpt_jacobian(helix).T @ forward_jacobian(helix).T)
                #* Transform the predicted state and covariance from [d1, phi0, omega~, z0, cotTheta] -> [d0, phi0, omega, z0, theta]
                helix.update_state(parameters_transformation(state_predicted, omega_sign, 1))
                P_back = backward_jacobian(helix) @ P_predicted @ backward_jacobian(helix).T

                #* Check if the eigenvalues of the covariance matrix are positive
                if np.any(np.linalg.eigvals(P_back) <= 0):
                    errors.bit_3 = 1
                    break

                #* Predicted hit and detector position
                predicted_hit, position_at_detector = collision_detection(helix, rtree, delaunay, position_at_detector)

                #* Check if the particle hit the detector
                if (position_at_detector is None) and spacepoint_measurements.shape[0] > 0:
                    errors.bit_4 = 1
                    break

                #* Find the measurement in the detector
                spacepoint_measurements, measured_hit = choose_spacepoint(spacepoint_measurements, position_at_detector)

                #* Check if we have a measurement
                if measured_hit is None:
                    errors.bit_5 = 1
                    break

                #* Update the arc length of the helix
                helix.update_arclength(measured_hit[4])

                #* Check if the measurement is inside the error ellipse
                is_inside = hit_miss(measured_hit[2:], predicted_hit, helix, P_back, position_at_detector[0], helix.get_arclength(), confidence_level=0.95)
                if is_inside is False:
                    errors.bit_6 = 1
                    break

                #* Update the Kalman filter with the measurement
                state_pre_update = parameters_transformation(helix.get_state(), omega_sign, 0)
                P_pre_update = forward_jacobian(helix) @ P_back @ forward_jacobian(helix).T
                #* Kalman filter update step
                state_updated, P_updated, K = kf_update(state_pre_update, P_pre_update, np.array([np.atan2(measured_hit[3], measured_hit[2]), measured_hit[4]]).T, get_H(np.linalg.norm(measured_hit[2:4]), helix.get_arclength()), R)
                #* Transform the updated state and covariance from [[d1, phi0, omega~, z0, cotTheta] -> [d0, phi0, omega, z0, theta]
                helix.update_state(parameters_transformation(state_updated, omega_sign, 1))
                P_cov = omega_jacobian(helix) @ backward_jacobian(helix) @ P_updated @ backward_jacobian(helix).T @ omega_jacobian(helix).T

                rows_to_write.append([event_number, particle_id, 0] + 
                                    helix.get_state().flatten().tolist() +
                                    P_cov.flatten().tolist() +
                                    K.flatten().tolist())

            #* If there are any errors, log them
            if len(errors.list_active_flags()) > 0:
                logger.debug(errors.to_json())
                continue
            
            #* Write the rows to the output file
            with open(output, mode="a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(rows_to_write)

