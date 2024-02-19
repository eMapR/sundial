import os
import logging


def get_logger(log_path: str, name: str) -> logging.Logger:
    """
    Get a simple logger instance with the specified log path.

    Args:
        log_path (str): The path where the log file will be saved. Defaults to None.

    Returns:
        logging.Logger: The logger instance.
    """
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_path, name + '.log'))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
