import os
import logging


def get_logger(log_path: str, name: str = __name__) -> logging.Logger:
    """
    Get a simple logger instance with the specified log path.

    Args:
        log_path (str): The path where the log file will be saved. Defaults to None.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.join(log_path, __name__ + '.log'))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
