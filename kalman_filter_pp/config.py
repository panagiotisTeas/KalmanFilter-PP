import logging

LOG_NAME = "kalman_filter_pp"
LOG_FORMAT = "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s"
ENABLE_FILE_LOGGING = True
ENABLE_CONSOLE_LOGGING = True
LOG_FILE_PATH = "output/logs/kalman_filter.log"
LOG_LEVEL = logging.DEBUG