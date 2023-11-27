import logging
from pathlib import Path


def setup_logger(log_file):
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)

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
