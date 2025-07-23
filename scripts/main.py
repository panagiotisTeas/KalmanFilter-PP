import numpy as np
import pandas as pd
import logging

from kalman_filter_pp.algorithms import linear_kalman_algorithm
from kalman_filter_pp.logger import set_log_level

set_log_level(logging.INFO)

linear_kalman_algorithm()

