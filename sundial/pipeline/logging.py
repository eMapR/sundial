import os
import logging
import time

from constants import LOG_PATH, METHOD


def get_logger(log_path: str, name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    os.makedirs(log_path, exist_ok=True)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_path, name + '.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger


def function_timer(func):
    logger = get_logger(LOG_PATH, METHOD)

    def timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if logger:
            logger.info(f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
        else:
            print(f"{func.__name__} took {(end - start)/60:.3f} minutes to complete.")
        return result
    return timer
