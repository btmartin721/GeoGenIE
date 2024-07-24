import logging
from pathlib import Path

from geogenie import GeoGenIE
from geogenie.utils.argument_parser import setup_parser
from geogenie.utils.logger import setup_logger


def main(args):
    logfile = Path(args.output_dir, "logfiles", f"{args.prefix}_logfile.txt")
    setup_logger(str(logfile), log_level=logging.DEBUG)
    genie = GeoGenIE(args)

    genie.train_test_predict()


if __name__ == "__main__":
    main(setup_parser())
