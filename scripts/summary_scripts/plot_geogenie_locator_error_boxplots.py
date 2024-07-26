import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser(
        "summarize_metrics_boot_combos2.py",
        description="Summarize metrics across all bootstrap replicates for both GeoGenIE and Locator.",
    )

    parser.add_argument(
        "-g",
        "--geogenie_dir",
        required=True,
        type=str,
        help="Specify parent directory where all GeoGenIE results are stored.",
    )
    parser.add_argument(
        "-l",
        "--locator_dir",
        required=True,
        type=str,
        help="Specify parent directory where all Locator results are stored.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        type=str,
        help="Specify directory where results from this script will be stored.",
    )
    parser.add_argument(
        "-c",
        "--coords_file",
        required=True,
        type=str,
        help="CSV or TSV file with sampleID as the first column, 'x' (longitude) as the second, and 'y' (latitude) as the third.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        type=str,
        help="Prefix to use for output files.",
    )

    return parser.parse_args()


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax = sns.boxplot()


if __name__ == "__main__":
    main(parse_arguments())
