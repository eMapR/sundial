import os
import logging

LOGGERS = {}


def get_logger(log_path: str, name: str) -> logging.Logger:
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if LOGGERS.get(name):
        return LOGGERS.get(name)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(log_path, name + '.log'))
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    LOGGERS[name] = logger

    return logger
