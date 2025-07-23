import numpy as np
import pandas as pd
import logging

from kalman_filter_pp.algorithms import linear_kalman_algorithm
from kalman_filter_pp.logger import set_log_level
from kalman_filter_pp.dists import plot_residual_distribution, plot_pull_distribution
from kalman_filter_pp.evos import plot_column_evolution, plot_fit_parameter_evolution
from kalman_filter_pp.config import KALMAN_FILTER_EXPORT_PATH

# set_log_level(logging.INFO)

# linear_kalman_algorithm()

plot_residual_distribution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH))
plot_pull_distribution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH))
plot_fit_parameter_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448)
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P00")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P11")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P22")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P33")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P44")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P01")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P02")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P12")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "P34")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "K00")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "K10")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "K20")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "K31")
plot_column_evolution(pd.read_csv(KALMAN_FILTER_EXPORT_PATH), 1000, 4507516637544448, "K41")
