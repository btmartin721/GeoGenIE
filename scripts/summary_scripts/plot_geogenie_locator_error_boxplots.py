import argparse
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from geogenie.utils.utils import read_csv_with_dynamic_sep
from geogenie.plotting.plotting import PlotGenIE


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


def read_and_combine_csv_data(directories, suffix="*_test_predictions.csv"):
    """
    Read CSV files from given directories and combine them into a single DataFrame.

    Args:
        directories (list of str): List of directory names to read the CSV files from.

    Returns:
        pd.DataFrame: Combined DataFrame with all the runtime data.
    """
    dataset_dfs = []
    for directory in directories:
        all_data = []
        pth = Path(directory)
        if (
            "test" in pth.name.lower()
            or "val" in pth.name.lower()
            or "unknown" in pth.name.lower()
        ):
            dataset = pth.name
            print(f"Loading {dataset} dataset...")
        else:
            print(
                "Path does not contain dataset name. Assuming this is the 'test' dataset."
            )
            dataset = "test"

        if not pth.exists() or not pth.is_dir():
            raise NotADirectoryError(
                f"Directory {directory} either not found or it is not a directory."
            )

        if pth.name.endswith("bootFULL_predlocs.txt") or pth.name.endswith(
            "bootFULL_testlocs.txt"
        ):
            print(f"Skipping directory: {pth.name}")
            continue

        for csv_path in Path(directory).glob(suffix):
            temp_df = read_csv_with_dynamic_sep(csv_path)

            if "Unnamed: 0" in temp_df.columns:
                temp_df = temp_df.drop(labels=["Unnamed: 0"], axis=1)

            if temp_df.shape[1] > 3:
                raise ValueError(
                    f"Invalid number of columns in loaded bootstrap prediction files. Expected three columns, got: {temp_df.shape[1]}. Column names: {temp_df.columns}"
                )

            for col in temp_df.columns:
                if col not in {"sampleID", "x", "y", "x_mean", "y_mean"}:
                    raise ValueError(
                        f"Invalid columns present in bootstrap prediction files. Supported columns include only: 'sampleID', 'x', 'y', but got: {col}"
                    )
            all_data.append(temp_df)
        all_data_df = pd.concat(all_data)
        all_data_df["dataset"] = dataset
        dataset_dfs.append(all_data_df)

    all_datasets_df = pd.concat(dataset_dfs)

    if all_datasets_df.empty:
        raise ValueError("Predictions not loaded. Output DataFrame was empty.")

    return all_datasets_df


def setup_output_directories(output_dir):
    """Sets up output directories for storing plots and shapefiles."""
    outdir = Path(output_dir)
    Path(outdir, "plots", "shapefile").mkdir(exist_ok=True, parents=True)
    return outdir


def process_known_data(coords_file):
    """Reads and processes the known coordinates file."""
    df_known = read_csv_with_dynamic_sep(coords_file)
    df_known = df_known.dropna(axis=0, how="any").sort_values(by="sampleID")
    return df_known


def process_software_data(directory, suffix, dataset_name):
    """Reads and processes software-specific data."""
    data_dir = Path(directory)
    df = read_and_combine_csv_data([data_dir], suffix=suffix)

    if df.empty:
        raise ValueError(f"No data found in directory: {data_dir}")

    if dataset_name.lower() == "geogenie":
        dataset_name = "GeoGenIE (Best Model)"
    elif dataset_name.lower() == "locator":
        dataset_name = "Locator"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    df["dataset"] = dataset_name
    return df.sort_values(by="sampleID")


def synchronize_sampleIDs(df_known, dfgg, dfloc):
    """
    Synchronize df_known with dfgg and dfloc by retaining only sampleIDs
    present in both dfgg and dfloc. Includes validation for missing sampleIDs.

    Args:
        df_known (pd.DataFrame): Known data with unique sampleIDs.
        dfgg (pd.DataFrame): GeoGenIE bootstrap replicate data.
        dfloc (pd.DataFrame): Locator bootstrap replicate data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Synchronized DataFrames.
    """
    # Extract unique sampleIDs from dfgg and dfloc
    gg_sampleIDs = set(dfgg["sampleID"].unique())
    loc_sampleIDs = set(dfloc["sampleID"].unique())

    # Find common sampleIDs in both dfgg and dfloc
    common_sampleIDs = gg_sampleIDs & loc_sampleIDs

    # Validate for missing sampleIDs in df_known
    missing_in_known = common_sampleIDs - set(df_known["sampleID"].unique())
    if missing_in_known:
        print(
            f"Warning: The following sampleIDs are in dfgg/dfloc but missing in df_known: {missing_in_known}"
        )

    # Filter df_known to retain only common sampleIDs
    df_known = df_known[df_known["sampleID"].isin(common_sampleIDs)]

    # Filter dfgg and dfloc to retain rows with sampleIDs in df_known
    dfgg = dfgg[dfgg["sampleID"].isin(df_known["sampleID"])]
    dfloc = dfloc[dfloc["sampleID"].isin(df_known["sampleID"])]

    # Ensure columns are in the same order
    dfloc = dfloc[dfgg.columns]

    return df_known, dfgg, dfloc


def setup_plots(outdir, software, df, df_known, gray_counties):
    """Handles plotting for each dataset."""
    dataset_dir = outdir / software / "test"
    shpdir = dataset_dir / "plots" / "shapefile"
    Path(shpdir).mkdir(exist_ok=True, parents=True)

    print(f"Saving {software} test set plots to: {dataset_dir}")

    plotting = PlotGenIE(
        device="cpu",
        output_dir=dataset_dir,
        prefix=f"{software}_bootstrap",
        basemap_fips="05",
        show_plots=False,
        basemap_highlights=gray_counties,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip",
        filetype="pdf",
        dpi=300,
        fontsize=24,
        remove_splines=True,
    )

    # Ensure correct CRS.
    gdf = plotting.processor.to_geopandas(df)
    df = plotting.processor.to_pandas(gdf)
    gdfk = plotting.processor.to_geopandas(plotting.processor.to_geopandas(df_known))
    dfk = plotting.processor.to_pandas(gdfk)

    return plotting, df, dfk


def main(args):

    # Setup directories
    outdir = setup_output_directories(args.output_dir)

    # Define counties for highlighting
    gray_counties = "Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley"

    ggdir = Path(args.geogenie_dir) / "bootstrap_predictions" / "test"

    # Process data
    df_known = process_known_data(args.coords_file)
    dfgg = process_software_data(ggdir, "*_test_predictions.csv", "geogenie")
    dfloc = process_software_data(args.locator_dir, "*locs.txt", "locator")

    # Synchronize sampleIDs across DataFrames
    df_known, dfgg, dfloc = synchronize_sampleIDs(df_known, dfgg, dfloc)

    plotter, dfgg, dfk = setup_plots(outdir, "geogenie", dfgg, df_known, gray_counties)

    _, dfloc, __ = setup_plots(outdir, "locator", dfloc, df_known, gray_counties)

    gdfgg = plotter.processor.to_geopandas(dfgg)
    gdfloc = plotter.processor.to_geopandas(dfloc)

    gdfgg_grp = gdfgg.dissolve(
        by="sampleID", aggfunc={"x": [np.mean, np.median], "y": [np.mean, np.median]}
    ).reset_index()

    # Flatten MultiIndex columns
    gdfgg_grp.columns = [
        "_".join(col) if isinstance(col, tuple) else col for col in gdfgg_grp.columns
    ]

    gdfloc_grp = gdfloc.dissolve(
        by="sampleID", aggfunc={"x": [np.mean, np.median], "y": [np.mean, np.median]}
    ).reset_index()

    gdfgg = gdfgg_grp.copy()
    gdfloc = gdfloc_grp.copy()

    gdfloc.columns = [
        "_".join(col) if isinstance(col, tuple) else col for col in gdfloc_grp.columns
    ]

    haversine_locator = plotter.processor.haversine_distance(
        gdfloc[["x_mean", "y_mean"]], dfk[["x", "y"]]
    )

    haversine_geogenie = plotter.processor.haversine_distance(
        gdfgg[["x_mean", "y_mean"]], dfk[["x", "y"]]
    )

    haversine_locator_median = plotter.processor.haversine_distance(
        gdfloc[["x_median", "y_median"]], dfk[["x", "y"]]
    )

    haversine_geogenie_median = plotter.processor.haversine_distance(
        gdfgg[["x_median", "y_median"]], dfk[["x", "y"]]
    )

    gdfloc["haversine_error"] = haversine_locator
    gdfgg["haversine_error"] = haversine_geogenie
    gdfloc["haversine_error_median"] = haversine_locator_median
    gdfgg["haversine_error_median"] = haversine_geogenie_median

    gdfloc["dataset"] = "Locator"
    gdfgg["dataset"] = "GeoGenIE"

    df = pd.concat([gdfloc, gdfgg])

    df_melt = df.melt(
        id_vars=[
            "sampleID",
            "dataset",
            "x_mean",
            "y_mean",
            "x_median",
            "y_median",
            "geometry",
            "haversine_error_median",
        ],
        value_vars=["haversine_error"],
        var_name="variable",
        value_name="Mean Prediction Error (km)",
    )

    desc = df_melt.groupby(["dataset"])["Mean Prediction Error (km)"].describe()
    desc.to_csv(outdir / "geogenie_locator_mean_error_summary.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    ax = axs[0]

    sns.set_theme(style="white")

    ax = sns.boxplot(
        x="dataset",
        y="Mean Prediction Error (km)",
        hue="dataset",
        data=df_melt,
        palette="Set1",
        ax=ax,
    )

    ax.set_title("Prediction Error Comparison", fontsize=24)
    ax.set_ylabel("Mean Prediction Error (km)", fontsize=24)
    ax.set_xlabel("Model", fontsize=24)
    ax.set_ylim([0, 300])

    ax = axs[1]

    df_melt2 = df.melt(
        id_vars=[
            "sampleID",
            "dataset",
            "x_mean",
            "y_mean",
            "x_median",
            "y_median",
            "geometry",
            "haversine_error",
        ],
        value_vars=["haversine_error_median"],
        var_name="variable",
        value_name="Median Prediction Error (km)",
    )

    ax = sns.boxplot(
        x="dataset",
        y="Median Prediction Error (km)",
        hue="dataset",
        data=df_melt2,
        palette="Set1",
        ax=ax,
    )

    ax.set_title("Prediction Error Comparison", fontsize=24)
    ax.set_ylabel("Median Prediction Error (km)", fontsize=24)
    ax.set_xlabel("Model", fontsize=24)
    ax.set_ylim([0, 300])

    fig.savefig(
        outdir / "geogenie_locator_mean_median_error_boxplot.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close()

    desc = df_melt2.groupby(["dataset"])["Median Prediction Error (km)"].describe()
    desc.to_csv(outdir / "geogenie_locator_median_error_summary.csv")


if __name__ == "__main__":
    main(parse_arguments())
