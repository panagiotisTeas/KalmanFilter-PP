import os
import logging

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

LOG_NAME = "kalman_filter_pp"
LOG_FORMAT = "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
ENABLE_FILE_LOGGING = True
ENABLE_CONSOLE_LOGGING = True
LOG_FILE_PATH = os.path.join(BASE_DIR, "output/logs/kalman_filter.log")
LOG_LEVEL = logging.DEBUG

DETECTOR_IMPORT_PATH = os.path.join(BASE_DIR, "data/processed/")
PARTICLE_IMPORT_PATH = os.path.join(BASE_DIR, "data/processed/")