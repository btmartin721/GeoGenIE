import argparse
import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.logger import setup_logger
from geogenie.utils.utils import read_csv_with_dynamic_sep

fontsize = 24
dpi = 300

# Adjust matplotlib settings globally.
sizes = {
    "axes.labelsize": fontsize,
    "axes.titlesize": fontsize,
    "figure.titlesize": fontsize,
    "figure.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "legend.title_fontsize": fontsize,
    "figure.dpi": dpi,
    "savefig.dpi": dpi,
}

sns.set_context("paper", rc=sizes)
plt.rcParams.update(sizes)
mpl.rcParams.update(sizes)


def read_and_combine_csv_data(directories, suffix="*_test_predictions.csv"):
    """
    Read CSV files from given directories and combine them into a single DataFrame.

    Args:
        directories (list of str): List of directory names to read the CSV files from.

    Returns:
        pd.DataFrame: Combined DataFrame with all the runtime data.
    """
    if not suffix.startswith("*"):
        suffix = "*" * suffix

    all_data = pd.DataFrame()
    for directory in directories:
        for csv_path in Path(directory).glob(suffix):
            temp_df = read_csv_with_dynamic_sep(csv_path)
            all_data = pd.concat([all_data, temp_df], ignore_index=True)

    if all_data.empty:
        raise ValueError("Predictions not loaded correctly.")
    return all_data


def sample_group(group):
    """Sample 100 entries from a group if the group size is greater than 100, or return the entire group otherwise.

    Args:
        group (DataFrame): The group of data from a DataFrame.

    Returns:
        DataFrame: A sampled subset of the group if its size is over 100, or the full group if 100 or fewer.
    """
    if len(group) > 100:
        return group.sample(n=100)  # Randomly sample 100 entries from the group
    return group  # Return the full group if it has 100 or fewer entries


def grouped_ci_boot(
    plotting,
    logger,
    df,
    df_known,
    model,
    dataset,
    output_dir,
    debug=False,
):
    """
    Process locality data for each sample to calculate mean, standard deviation, confidence intervals, and DRMS (Distance Root Mean Square).

    Args:
        plotting (PlotGenIE): PlotGenIE object to use.
        logger (logging.logger): Logger object to use.
        df (pd.DataFrame): DataFrame containing 'x', 'y' coordinates and 'sampleID' column.
        df_known (pd.DataFrame): DataFrame containing true 'x', 'y' coordinates and 'sampleID' column.
        model (str): Model to use. Should be one of: 'geogenie' or 'locator'.
        dataset (str): Which dataset is being used. Should be one of {"test", "val", "pred"}.
        output_dir (str): Directory to output plots to.

    Returns:
        pd.DataFrame: DataFrame with calculated statistics for each sample.
    """

    logger.info(f"Generating and saving {dataset} CI plots to: {output_dir}")

    # If > 100 bootstraps, randomly sample 100 bootstrap replicates.
    sample_df = df.groupby("sampleID").apply(sample_group).reset_index(drop=True)

    gdf = plotting.processor.to_geopandas(sample_df)

    if df_known is None and dataset != "pred":
        logger.info("Known coordinates were not provided.")

    if model != "locator" and model != "geogenie":
        raise ValueError("Invalid model argument provided.")

    results = []
    groups = []
    for i, (group, sample_id, dfk, resd) in enumerate(
        plotting.processor.calculate_statistics(gdf, known_coords=df_known)
    ):
        results.append(resd)

        grp = plotting.processor.to_pandas(group)

        if debug:
            print(grp)
            print(dfk)

        grp["x_true"] = dfk["x"].iloc[0]
        grp["y_true"] = dfk["y"].iloc[0]

        dfgrp = pd.DataFrame(
            {
                "haversine_error": plotting.processor.haversine_distance(
                    grp[["x", "y"]].to_numpy(), grp[["x_true", "y_true"]].to_numpy()
                )
            }
        )

        dfgrp["sampleID"] = sample_id
        dfgrp["bootrep"] = i

        groups.append(dfgrp)

    dfres = pd.DataFrame(results)
    dfgroups = pd.concat(groups)
    return dfres, dfgroups


def parse_filename(filename):
    """
    Extract 'out', 'wt', and 'meth' parameters from the filename.
    Raises a ValueError if any parameter cannot be found.
    """
    parts = filename.stem.split("_")

    try:
        out_index = parts.index("out")
        out_value = parts[out_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'out' value not found in filename: {filename}")

    try:
        wt_index = parts.index("wt")
        wt_value = parts[wt_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'wt' value not found in filename: {filename}")

    try:
        meth_index = parts.index("meth")
        meth_value = parts[meth_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'meth' value not found in filename: {filename}")

    return f"{wt_value}_{out_value}_{meth_value}"


def update_metric_labels(df):
    metric_map = {
        "mean_dist": "Mean Error",
        "median_dist": "Median Error",
        "stdev_dist": "StdDev of Error",
        "std_dist": "StdDev of Error",
    }
    df["Metric"] = df["Metric"].map(metric_map)
    return df


def calculate_correlation_for_plot(
    df, x_column, y_column, absent_val, present_val, stat
):
    filtered_df = df[(df[x_column] == absent_val) | (df[x_column] == present_val)]
    filtered_df = filtered_df[filtered_df["Metric"] == stat]
    pivot_df = filtered_df.pivot(columns=x_column, values=y_column)
    absent_data, present_data = pivot_df[absent_val], pivot_df[present_val]
    return pearsonr(absent_data, present_data)


def filter_configs(df, include_weighted_sampler):
    """
    Filters the DataFrame based on the inclusion or exclusion of 'GeoGenIE: Weighted Sampler'.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        include_weighted_sampler (bool): Whether to include entries with 'GeoGenIE: Weighted Sampler'.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if include_weighted_sampler:
        return df
    else:
        return df[~df["config"].str.contains("Weighted Sampler")]


def create_facet_grid(df, outpath, include_weighted_sampler=True, ft="pdf"):
    df = filter_configs(df, include_weighted_sampler)
    fig, axs = plt.subplots(1, 1, figsize=(24, 10))
    ax1 = axs

    df1 = df[["mean_dist", "median_dist", "stdev_dist", "std_dist", "config"]]
    df_melted1 = df1.melt(var_name="Metric", value_name="Value", id_vars=["config"])
    df_melted1 = update_metric_labels(df_melted1)
    df_melted1 = df_melted1.sort_values(by=["Metric", "config"], ascending=True)

    # # Specify the order of the legend items
    # hue_order = [
    #     "Original Locator",
    #     "No Sample Weights",
    #     "Weighted Loss",
    #     "No Sample Weights + Outliers Removed",
    #     "Weighted Loss + Outliers Removed",
    #     "No Sample Weights + Oversampling",
    #     "Weighted Loss + Oversampling",
    #     "No Sample Weights + Outliers Removed + Oversampling",
    #     "Weighted Loss + Outliers Removed + Oversampling",
    # ]

    # sns.boxplot(
    #     data=df_melted1,
    #     x="Metric",
    #     y="Value",
    #     hue="config",
    #     hue_order=hue_order,
    #     ax=ax1,
    # )
    # ax1.legend(title="Model Configuration")
    # sns.move_legend(ax1, loc="center", bbox_to_anchor=(0.5, 1.2), ncol=2)
    # ax1.set_xlabel("Summary Statistic")
    # ax1.set_ylabel("Prediction Error (km)")
    # ax1.set_xticks(ticks=ax1.get_xticks(), labels=ax1.get_xticklabels())

    # fn = f"performance_summary_gg_locator_boxplots.{ft}"
    # outfile = outpath / fn

    # fig.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=dpi)
    # plt.close()
    # print(f"Performance summary boxplots saved to: {outfile}.")

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))

    # Specify the order of the legend items
    hue_order2 = ["Original Locator", "GeoGenIE (Best Model)"]

    means = (
        df_melted1[
            (df_melted1["config"].isin(hue_order2))
            & (df_melted1["Metric"] == "Mean Error")
        ]
        .groupby(["Metric", "config"])["Value"]
        .mean()
    )

    medians = (
        df_melted1[
            (df_melted1["config"].isin(hue_order2))
            & (df_melted1["Metric"] == "Median Error")
        ]
        .groupby(["Metric", "config"])["Value"]
        .median()
    )

    stddevs = (
        df_melted1[
            (df_melted1["config"].isin(hue_order2))
            & (df_melted1["Metric"] == "StdDev of Error")
        ]
        .groupby(["Metric", "config"])["Value"]
        .std()
    )

    print(f"Mean Error: {means}")
    print(f"Median Error: {medians}")
    print(f"StdDev Error: {stddevs}")

    sns.boxplot(
        data=df_melted1[df_melted1["config"].isin(hue_order2)],
        x="Metric",
        y="Value",
        hue="config",
        hue_order=hue_order2,
        ax=ax1,
    )
    ax1.legend(title="Model Configuration")
    sns.move_legend(ax1, loc="center", bbox_to_anchor=(0.5, 1.1))
    ax1.set_xlabel("Summary Statistic")
    ax1.set_ylabel("Prediction Error (km)")
    ax1.set_xticks(ticks=ax1.get_xticks(), labels=ax1.get_xticklabels())

    fn = f"performance_summary_gg_locator_boxplots_best_model.{ft}"
    outfile = outpath / fn

    fig.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Performance summary boxplots (best model) saved to: {outfile}.")

    fig, axs = plt.subplots(1, 3, figsize=(24, 10))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    df_melted1["Sample Weighting"] = "Absent"

    df_melted1.loc[
        df_melted1["config"].str.startswith("Weighted Loss"), "Sample Weighting"
    ] = "Present"

    df_melted1["Outliers"] = "Absent"

    df_melted1.loc[
        ~df_melted1["config"].str.contains("Outliers Removed"), "Outliers"
    ] = "Present"

    df_melted1["Oversampling"] = "Absent"

    df_melted1.loc[
        ~df_melted1["config"].str.contains("Oversampling"), "Oversampling"
    ] = "Present"

    hue_order = ["Mean Error", "Median Error", "StdDev of Error"]

    sns.lineplot(
        data=df_melted1[~df_melted1["config"].str.contains("Original Locator")],
        x="Sample Weighting",
        y="Value",
        hue="Metric",
        hue_order=hue_order,
        ax=ax1,
    )

    ax1.legend(title="Model Configuration")
    sns.move_legend(ax1, loc="center", bbox_to_anchor=(0.5, 1.2))
    ax1.set_xlabel("Sample Weighting")
    ax1.set_ylabel("Prediction Error (km)")
    ax1.set_xticks(ticks=ax1.get_xticks(), labels=ax1.get_xticklabels())

    sns.lineplot(
        data=df_melted1[~df_melted1["config"].str.contains("Original Locator")],
        x="Outliers",
        y="Value",
        hue="Metric",
        hue_order=hue_order,
        ax=ax2,
    )

    ax2.legend(title="Model Configuration")
    sns.move_legend(ax2, loc="center", bbox_to_anchor=(0.5, 1.2))
    ax2.set_xlabel("Outliers")
    ax2.set_ylabel("Prediction Error (km)")
    ax2.set_xticks(ticks=ax2.get_xticks(), labels=ax2.get_xticklabels())

    sns.lineplot(
        data=df_melted1[~df_melted1["config"].str.contains("Original Locator")],
        x="Oversampling",
        y="Value",
        hue="Metric",
        hue_order=hue_order,
        ax=ax3,
    )

    ax3.legend(title="Model Configuration")
    sns.move_legend(ax3, loc="center", bbox_to_anchor=(0.5, 1.2))
    ax3.set_xlabel("Oversampling")
    ax3.set_ylabel("Prediction Error (km)")
    ax3.set_xticks(ticks=ax3.get_xticks(), labels=ax3.get_xticklabels())

    fn = f"performance_summary_gg_locator_lineplots.{ft}"
    outfile = outfile.with_name(fn)

    fig.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Performance summary lineplots saved to: {outfile}.")


def read_directory_of_files(directory_path, suffix="*_metrics.json"):
    if not directory_path.exists() or not directory_path.is_dir():
        raise NotADirectoryError(
            f"The specified directory does not exist: {str(directory_path)}"
        )

    data = []
    files = list(directory_path.glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(
            f"No '{suffix}' files found in directory: {str(directory_path)}"
        )

    for file_path in files:
        with file_path.open("r") as f:
            json_data = json.load(f)
            # json_data["config"] = parse_filename(file_path)
            data.append(json_data)
    df = pd.DataFrame(data)
    return df.dropna(how="any", axis=0)


def map_config_to_label(config):
    """
    Map the given config string to a more descriptive label.
    """
    configs = config.split("_")
    if len(configs) == 1:
        return configs[0]

    if len(configs) != 3:
        raise ValueError(f"Invalid value found in 'config' column: {configs}")

    weighted, outlier, oversamp = configs

    # Determine translocation status
    translocations_label = "Outliers Removed" if outlier == "true" else ""
    meth_label = "Oversampling" if oversamp == "kmeans" else ""

    # Determine sampler/loss status based on the first part
    if weighted == "sampler":
        weight_label = "Weighted Sampler"
    elif weighted == "loss":
        weight_label = "Weighted Loss"
    elif weighted == "none":
        weight_label = "No Sample Weights"
    else:
        raise ValueError(f"Unexpected configuration type: {weighted}")

    lab = (
        f"{weight_label} + {translocations_label}"
        if translocations_label != ""
        else f"{weight_label}"
    )

    lab = f"{lab} + {meth_label}" if meth_label != "" else f"{lab}"
    return lab


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
        type=str,
        required=True,
        help="Comma-delimited or tab-delimited oordinates file with known coordinates for each sample in prediction/ test/ validation datasets.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        required=True,
        help="Prefix to use for output files.",
    )

    return parser.parse_args()


def main(args):

    # Set up the logfile.
    setup_logger(Path(args.output_dir) / f"{args.prefix}_logfile.txt")
    logger = logging.getLogger(__name__)

    gg_pth = Path(args.geogenie_dir) / "bootstrap_metrics" / "test"
    loc_pth = Path(args.locator_dir) / "test"

    df_gg = read_csv_with_dynamic_sep(
        gg_pth / "wtd_gtseq_N1415_L436_seed5238_N1415_L436_bootstrap_test_metrics.csv"
    )
    df_gg["config"] = "GeoGenIE (Best Model)"

    df_loc = read_and_combine_csv_data([loc_pth], suffix="*_predlocs.txt")
    df_loc = df_loc.sort_values(by=["sampleID"], ascending=True)
    df_loc["config"] = "Original Locator"

    df_known = read_csv_with_dynamic_sep(args.coords_file)
    df_known2 = df_known.copy()
    uniq_sids = pd.Series(df_loc["sampleID"].unique()).sort_values(ascending=True)
    df_known = df_known[df_known["sampleID"].isin(uniq_sids)]
    df_known = df_known.sort_values(by=["sampleID"], ascending=True)

    # Ensure in same order with same sampleIDs.
    if not uniq_sids.equals(df_known["sampleID"]):
        for row, uniq_row in zip(df_known["sampleID"].to_numpy().tolist(), uniq_sids):
            if row != uniq_row:

                raise ValueError(
                    f"'locator' and 'coords_file' sampleIDs could not be aligned: df_loc: {row}, df_known: {uniq_row}."
                )

    gray_counties = "Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley".split(
        ","
    )
    Path(args.output_dir, "plots", "shapefile").mkdir(exist_ok=True, parents=True)

    plotting = PlotGenIE(
        device="cpu",
        output_dir=args.output_dir,
        prefix=args.prefix,
        basemap_fips="05",
        basemap_highlights=gray_counties,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        show_plots=False,
        fontsize=24,
        filetype="pdf",
        dpi=300,
        remove_splines=True,
    )

    # Ensure correct CRS.
    gdf_loc = plotting.processor.to_geopandas(df_loc)
    df_loc = plotting.processor.to_pandas(gdf_loc)
    gdf_known = plotting.processor.to_geopandas(df_known)
    df_known = plotting.processor.to_pandas(gdf_known)

    _, dfgroups = grouped_ci_boot(
        plotting, logger, df_loc, df_known, "locator", "test", args.output_dir
    )

    df_loc = dfgroups.copy()
    df_loc["mean_dist"] = df_loc.groupby("bootrep")["haversine_error"].mean()
    df_loc["median_dist"] = df_loc.groupby("bootrep")["haversine_error"].median()
    df_loc["std_dist"] = df_loc.groupby("bootrep")["haversine_error"].std()

    logger.info(
        f"Locator Mean Validation Error for Haversine Distance: {dfgroups['haversine_error'].mean()}"
    )
    logger.info(
        f"Locator Median Validation Error for Haversine Distance: {dfgroups['haversine_error'].median()}"
    )
    logger.info(
        f"Locator Mean Validation Error for Haversine Distance: {dfgroups['haversine_error'].std()}"
    )

    df_loc["config"] = "Original Locator"

    df_gg["config"] = "GeoGenIE (Best Model)"

    df_combined = pd.concat([df_gg, df_loc])

    # df_combined["config"] = df_combined["config"].apply(map_config_to_label)

    outpth = Path(args.output_dir) / "plots"
    Path(outpth).mkdir(exist_ok=True, parents=True)
    create_facet_grid(df_combined, outpth, include_weighted_sampler=False)


if __name__ == "__main__":
    main(parse_arguments())
