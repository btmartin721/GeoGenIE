import os

from geogenie import GeoGenIE
from geogenie.utils.argument_parser import setup_parser
from geogenie.utils.logger import setup_logger


def main():
    args = setup_parser()

    logfile = os.path.join(
        args.output_dir,
        "logfiles",
        f"{args.prefix}_logfile.txt",
    )
    setup_logger(logfile)
    genie = GeoGenIE(args)

    genie.train_test_predict()


if __name__ == "__main__":
    main()
