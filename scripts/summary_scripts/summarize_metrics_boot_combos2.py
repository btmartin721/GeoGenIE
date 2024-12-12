import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from geogenie.plotting.plotting import PlotGenIE
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


def read_and_combine_csv_data(
    directories, suffix="*_test_predictions.csv", is_geogenie=True
):
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
        counter = 0
        for csv_path in Path(directory).glob(suffix):
            temp_df = read_csv_with_dynamic_sep(csv_path)

            if is_geogenie:
                temp_df["config"] = parse_filename(csv_path)
            else:
                temp_df["config"] = "Locator"
                temp_df["BootRep"] = counter
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
            counter += 1

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
    """Extract 'outlier', 'weighted', and 'oversamp' parameters from the filename.

    Args:
        filename (Path): The filename to parse.

    Returns:
        str: The configuration string in the format: "{oversamp_value}_{weighted_value}_{out_value}".

    Raises:
        ValueError: If any of the parameters cannot be found.
    """
    parts = filename.stem.split("_")

    try:
        out_index = parts.index("outlier")
        out_value = parts[out_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'outlier' value not found in filename: {filename}")

    try:
        weighted_index = parts.index("weighted")
        weighted_value = parts[weighted_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'weighted' value not found in filename: {filename}")

    try:
        oversamp_index = parts.index("oversample")
        oversamp_value = parts[oversamp_index + 1]
    except (ValueError, IndexError):
        raise ValueError(f"'oversample' value not found in filename: {filename}")

    return f"{oversamp_value}_{weighted_value}_{out_value}"


def map_config_to_label(config):
    """
    Map the configuration string to a more descriptive legend label.

    Args:
        config (str): Configuration string in the format: "{oversamp_value}_{weighted_value}_{outlier_value}".

    Returns:
        str: Human-readable description of the configuration. For example, "Oversampling, Loss Weighting, Outlier Detection".

    Raises:
        ValueError: If the configuration string is not in the expected format.
    """
    parts = config.split("_")
    if len(parts) != 3 and parts[0] != "Locator":
        raise ValueError(f"Invalid configuration format: {config}")

    if parts[0] == "Locator":
        return "Locator"

    oversamp, weighted, outlr = parts

    # Translate each part into a descriptive phrase
    oversamp_lab = "Oversampling" if oversamp == "kmeans" else "No Oversampling"
    wt_lab = "Weighted Loss" if weighted == "loss" else "No Weighted Loss"
    out_lab = "Outlier Detection" if outlr == "true" else "No Outlier Detection"

    # Combine all labels into a descriptive string
    label = f"{oversamp_lab}, {wt_lab}, {out_lab}"
    return label


def update_metric_labels(df):
    """Update the metric labels to be more descriptive.

    Args:
        df (pd.DataFrame): DataFrame with the metric labels to update.

    Returns:
        pd.DataFrame: DataFrame with updated metric labels.
    """
    metric_map = {
        "mean_dist": "Mean Error",
        "median_dist": "Median Error",
        "stdev_dist": "StdDev of Error",
        "std_dist": "StdDev of Error",
    }
    df["Metric"] = df["Metric"].map(metric_map)
    return df


def filter_configs(df, include_weighted_sampler):
    """Filters the DataFrame based on the inclusion or exclusion of 'GeoGenIE: Weighted Sampler'.

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
    """
    Create facet grid plots for the top 5 and bottom 5 models for each error statistic,
    with descriptive legend labels.

    Args:
        df (pd.DataFrame): Combined DataFrame with performance metrics.
        outpath (Path): Path to save plots.
        include_weighted_sampler (bool): Whether to include entries with "Weighted Sampler".
        ft (str): File type for saved plots (e.g., "pdf", "png").
    """
    df = filter_configs(df, include_weighted_sampler)

    # Define the order of the hue for the boxplots
    hue_order = [
        "Locator" "No Oversampling, No Weighted Loss, No Outlier Detection",
        "No Oversampling, No Weighted Loss, Outlier Detection",
        "No Oversampling, Weighted Loss, No Outlier Detection",
        "No Oversampling, Weighted Loss, Outlier Detection",
        "Oversampling, No Weighted Loss, No Outlier Detection"
        "Oversampling, No Weighted Loss, Outlier Detection",
        "Oversampling, Weighted Loss, No Outlier Detection",
        "Oversampling, Weighted Loss, Outlier Detection",
    ]

    # Filter the DataFrame to include only the selected configurations
    df_filtered = df.copy()

    # Map configurations to descriptive labels
    df_filtered["config_label"] = df_filtered["config"].apply(map_config_to_label)

    df_filtered = df_filtered.drop(columns=["config"])

    # Melt the DataFrame for plotting
    df_melted = df_filtered.melt(
        var_name="Metric", value_name="Value", id_vars=["config_label"]
    )

    df_melted = df_melted[
        df_melted["Metric"].isin(["mean_dist", "median_dist", "stdev_dist", "std_dist"])
    ]

    df_melted = update_metric_labels(df_melted)
    df_melted = df_melted.sort_values(by=["Metric", "config_label"], ascending=True)

    df_grouped_desc = df_melted.groupby("config_label").describe()
    df_grouped_desc.to_csv(outpath / "geogenie_locator_grouped_desc.csv")

    import sys

    sys.exit(0)

    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(max(24, len(hue_order) * 1.5), 12))

    # Create boxplot
    ax = sns.boxplot(
        data=df_melted,
        x="Metric",
        y="Value",
        hue="config_label",
        ax=ax,
    )
    ax.legend(
        title="Model Configuration",
        bbox_to_anchor=(0.5, 1.2),
        loc="center",
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    ax.set_xlabel("Summary Statistic")
    ax.set_ylabel("Prediction Error (km)")
    ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), ha="right")

    # Save plot
    fn = f"performance_summary_boxplots.{ft}"
    outfile = outpath / fn
    fig.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=dpi)
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(max(24, 2 * 1.5), 12))

    df_melted2 = df_melted.copy()
    df_melted2 = df_melted2[
        df_melted2["config_label"].isin(
            ["Oversampling, Weighted Loss, Outlier Detection", "Locator"]
        )
    ]

    df_melted2.loc[df_melted2["config_label"] != "Locator", "config_label"] = (
        "GeoGenIE (Best Model)"
    )

    # Create boxplot
    ax = sns.boxplot(data=df_melted2, x="Metric", y="Value", hue="config_label", ax=ax)

    ax.set_xlabel("Summary Statistic", fontsize=24)
    ax.set_ylabel("Prediction Error (km)", fontsize=24)
    ax.set_xticks(ticks=ax.get_xticks(), labels=ax.get_xticklabels(), ha="right")
    ax.xaxis.set_tick_params(labelsize=24)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.legend(
        title="Model Configuration",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=24,
    )

    # Save plot
    fn = f"performance_summary_boxplots_best_geogenie.{ft}"
    outfile = outpath / fn
    fig.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=dpi)
    plt.close()

    print(f"Performance summary boxplots saved to: {outfile}.")


def parse_arguments():
    """Parse command-line arguments for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        "summarize_metrics_boot_combos2.py",
        description="Summarize metrics across all bootstrap replicates for GeoGenIE and Locator.",
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
        required=False,
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
    gg_pth = Path(args.geogenie_dir) / "bootstrap_metrics" / "test"
    loc_pth = Path(args.locator_dir)

    df_gg = read_and_combine_csv_data([gg_pth], suffix="*_test_metrics.csv")
    df_loc = read_and_combine_csv_data(
        [loc_pth], suffix="*_testlocs.txt", is_geogenie=False
    )

    df_loc = df_loc.sort_values(by=["sampleID"], ascending=True)

    df_known = read_csv_with_dynamic_sep(args.coords_file)
    df_known = df_known.sort_values(by=["sampleID"], ascending=True)

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

    # Ensure consistent CRS for geospatial operations
    gdf_known = plotting.processor.to_geopandas(df_known)
    gdf_loc = plotting.processor.to_geopandas(df_loc)

    # Convert to pandas DataFrames
    df_known = plotting.processor.to_pandas(gdf_known)
    df_loc = plotting.processor.to_pandas(gdf_loc)

    # Filter `df_known` to retain only matching indices
    df_known = df_known.reset_index()
    df_known["index"] = df_known["index"].astype(int)

    df_gg = df_gg.reset_index()
    df_gg["index"] = df_gg["index"].astype(int)

    # Merge `df_gg` with `df_known` based on index alignment
    df_known = df_known.merge(df_gg, on="index", how="left")

    # Ensure alignment of coordinates between `df_known` and bootstrap replicates in `df_loc`
    df_known = df_known[["sampleID", "x", "y", "config"]].set_index("sampleID")
    df_loc = df_loc[["sampleID", "x", "y"]].set_index("sampleID")

    # Join `df_loc` with `df_known` on `sampleID` to align coordinates
    df_loc = df_loc.join(df_known, how="inner", lsuffix="_loc", rsuffix="_known")

    # Handle missing configurations
    if df_loc["config"].isnull().any():
        print("Warning: Some samples in df_loc are missing configurations from df_gg.")
        df_loc = df_loc.dropna(subset=["config"])  # Drop rows with missing config

    # Calculate Haversine distances for each sample in `df_loc`
    df_loc["haversine_distance"] = plotting.processor.haversine_distance(
        df_loc[["x_loc", "y_loc"]].to_numpy(), df_loc[["x_known", "y_known"]].to_numpy()
    )
    # Step 1: Assign replicate indices to `df_loc`
    df_loc = df_loc.reset_index()  # Ensure we have access to the sampleID as a column
    df_loc["replicate"] = (
        df_loc.groupby("config").cumcount() % 100 + 1
    )  # Create replicate indices (1–100)

    # Step 2: Summarize `df_loc` for all bootstrap replicates
    # Group by `replicate` to produce one row per bootstrap replicate
    df_loc_summary = (
        df_loc.groupby("replicate")["haversine_distance"]
        .agg(mean_dist="mean", median_dist="median", stdev_dist="std")
        .reset_index()
    )

    # Step 3: Prepare `df_gg` with the necessary columns (800 rows expected)
    df_gg = df_gg[["config", "mean_dist", "median_dist", "stdev_dist"]]

    # Step 4: Validate row counts before concatenation
    assert len(df_gg) == 800, "df_gg row count is incorrect; expected 800 rows."
    assert (
        len(df_loc_summary) == 100
    ), "df_loc_summary row count is incorrect; expected 100 rows."

    # Step 5: Concatenate `df_gg` (800 rows) and `df_loc_summary` (100 rows)
    df_final = pd.concat([df_gg, df_loc_summary], ignore_index=True)

    # Step 6: Validate final row count
    assert len(df_final) == 900, "df_final row count is incorrect; expected 900 rows."

    df_final.loc[df_final["config"].isna(), "config"] = "Locator"
    df_final = df_final.drop(columns=["replicate"])

    # Save or further process df_final
    outpth = Path(args.output_dir) / "plots"
    Path(outpth).mkdir(exist_ok=True, parents=True)

    # Optionally create facet grid plots using `df_gg`
    create_facet_grid(df_final, outpth, include_weighted_sampler=True)


if __name__ == "__main__":
    main(parse_arguments())
