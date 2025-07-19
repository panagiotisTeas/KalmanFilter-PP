import kalman_filter_pp.errors as errors
from kalman_filter_pp.logger import get_logger, set_log_level
import logging


err = errors.ErrorFlags(0b0100)

logger = get_logger()

logger.info("Logger initialized successfully.")
logger.debug("Debugging information: %s", err.to_dict())
logger.warning("Active error flags: %s", err.list_active_flags())
logger.error("Error in processing: %s", err.to_json())

set_log_level(logging.WARNING)

logger.info("Logger initialized successfully.")
logger.debug("Debugging information: %s", err.to_dict())
logger.warning("Active error flags: %s", err.list_active_flags())
logger.error("Error in processing: %s", err.to_json())

