import logging
from pathlib import Path

from geogenie import GeoGenIE
from geogenie.utils.argument_parser import setup_parser
from geogenie.utils.logger import setup_logger


def main():
    args = setup_parser()

    logfile = Path(args.output_dir) / "logfiles" / f"{args.prefix}_logfile.txt"
    log_level = logging.DEBUG if args.debug else logging.INFO

    setup_logger(str(logfile), log_level=log_level)
    genie = GeoGenIE(args)

    genie.train_test_predict = genie.total_execution_time_decorator(
        genie.train_test_predict
    )
    genie.train_test_predict()


if __name__ == "__main__":
    main()
