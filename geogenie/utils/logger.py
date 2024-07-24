import logging
from pathlib import Path


def setup_logger(log_file, log_level=logging.INFO):
    """Function to set up a logger for logging info, warnings, errors, and debug prints.

    Args:
        log_file (str): Filename for log file.
        log_level (int): Log level to use. Should be either logging.INFO, logging.WARNING, logging.ERROR, or logging.DEBUG. These levels are converted to integers when called. Defaults to logging.INFO.
    """
    logger = logging.getLogger()  # Root logger
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    Path(log_file).parents[0].mkdir(parents=True, exist_ok=True)

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
