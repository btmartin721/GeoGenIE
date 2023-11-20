import logging
import sys
from contextlib import contextmanager


class StreamToLogger:
    """
    Custom stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


@contextmanager
def redirect_logging(logger):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
