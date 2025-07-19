import logging

from .config import LOG_NAME, LOG_FORMAT, ENABLE_FILE_LOGGING, ENABLE_CONSOLE_LOGGING, LOG_FILE_PATH, LOG_LEVEL

def get_logger(name : str = LOG_NAME) -> logging.Logger:
    """
    Create and configure a logger.
    
    Args:
        name (str): Name of the logger.
    
    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(LOG_FORMAT)

    if ENABLE_FILE_LOGGING:
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if ENABLE_CONSOLE_LOGGING:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def set_log_level(level: int) -> None:
    """
    Set global log level for all loggers.
    
    Args:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
    """

    logging.getLogger(LOG_NAME).setLevel(level)