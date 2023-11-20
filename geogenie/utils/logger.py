import logging


def setup_logger(log_file):
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

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
