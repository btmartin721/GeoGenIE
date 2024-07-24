import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.utils import read_csv_with_dynamic_sep


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
        help="Specify directory where all results from this script will be stored.",
    )
    parser.add_argument(
        "-c",
        "--coords_file",
        required=True,
        type=str,
        help="Specify file where coordinates data are stored. Should be a CSV or TSV file with sampleID as the first column, longitude ('x') as the second column, and latitude ('y') as the third column.",
    )

    return parser.parse_args()


def make_all_plots(plotting, df, dataset, df_known=None, seed=None):
    results = []
    results_known = []
    haversine_errs = []

    gdf = plotting.processor.to_geopandas(df)

    print("Generating per-sample contour plots...")

    for i, (group, sample_id, dfk, resd) in enumerate(
        plotting.processor.calculate_statistics(gdf, seed=seed, known_coords=df_known)
    ):

        dfk2 = df_known[df_known["sampleID"] == sample_id]
        gdfk2 = plotting.processor.to_geopandas(dfk2)
        dfk2 = plotting.processor.to_pandas(gdfk2)

        dfk2.columns = ["sampleID", "x", "y"]

        results.append(resd)

        df_known_res = dfk2.copy()

        mean_lon = group.dissolve().centroid.x
        mean_lat = group.dissolve().centroid.y

        haversine_error = plotting.processor.haversine_distance(
            np.array([[mean_lat, mean_lon]]),
            np.array([[df_known_res["y"].iloc[0], df_known_res["x"].iloc[0]]]),
        )

        haversine_errs.append(haversine_error)

        dfk3 = pd.DataFrame(df_known_res.to_numpy().tolist())
        results_known.append(dfk3)

    dferrs = np.array(np.squeeze(haversine_errs))
    df_known_res = pd.concat(results_known)
    dfres = pd.DataFrame(results)
    df_known_res.columns = ["sampleID", "x", "y"]

    if df_known is not None and not dfres.empty and not df_known.empty:

        print("Generating all plots...")

        plotting.plot_geographic_error_distribution(
            df_known_res[["x", "y"]].to_numpy(),
            dfres[["x_mean", "y_mean"]].to_numpy(),
            dataset,
            buffer=0.1,
            marker_scale_factor=2,
            min_colorscale=0,
            max_colorscale=300,
            n_contour_levels=20,
        )

        plotting.polynomial_regression_plot(
            df_known_res[["x", "y"]].to_numpy(),
            dfres[["x_mean", "y_mean"]].to_numpy(),
            dataset,
            max_ylim=300,
            max_xlim=2.0,
            n_xticks=5,
        )

        plotting.plot_cumulative_error_distribution(
            dferrs,
            f"{plotting.pfx}_cumulative_error_dist.{plotting.filetype}",
            np.percentile(dfres["haversine_error"], [25, 50, 75]),
            dfres["haversine_error"].median(),
            dfres["haversine_error"].mean(),
        )

        plotting.plot_error_distribution(
            dfres["haversine_error"].to_numpy(),
            str(plotting.obp)
            + f"{plotting.pfx}_error_distribution.{plotting.filetype}",
        )

        z_scores = (dfres["haversine_error"] - dfres["haversine_error"].mean()) / dfres[
            "haversine_error"
        ].std()

        plotting.plot_zscores(
            z_scores.to_numpy(), f"{plotting.pfx}_zscores.{plotting.filetype}"
        )
    else:
        raise pd.errors.EmptyDataError("Encountered empty DataFrame.")

    return dfres


def read_and_combine_csv_data(directories, suffix="*_test_predictions.csv"):
    """
    Read CSV files from given directories and combine them into a single DataFrame.

    Args:
        directories (list of str): List of directory names to read the CSV files from.

    Returns:
        pd.DataFrame: Combined DataFrame with all the runtime data.
    """
    sep = "," if suffix.endswith(".csv") else "\t"

    dataset_dfs = []
    for directory in directories:
        all_data = []
        pth = Path(directory)
        if pth.name in {"test", "val", "unknown"}:
            dataset = pth.name
            print(f"Loading {dataset} dataset.")
        else:
            print(
                "Path does not contain dataset name. Assuming this is the 'test' dataset."
            )
            dataset = "test"

        if not pth.exists() or not pth.is_dir():
            raise NotADirectoryError(
                f"Directory {directory} either not found or it is not a directory."
            )

        for csv_path in Path(directory).glob(suffix):
            temp_df = read_csv_with_dynamic_sep(csv_path)

            if "Unnamed: 0" in temp_df.columns:
                temp_df = temp_df.drop(labels=["Unnamed: 0"], axis=1)

            if temp_df.shape[1] > 3:
                raise ValueError(
                    f"Invalid number of columns in loaded bootstrap prediction files. Expected three columns, got: {temp_df.shape[1]}. Column names: {temp_df.columns}"
                )

            for col in temp_df.columns:
                if col not in {"sampleID", "x", "y"}:
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


def preprocess_known(true_df, d, df, plotting):
    df = df.copy()
    true_df = true_df.copy()
    gdf_known = plotting.processor.to_geopandas(true_df)
    df_known = plotting.processor.to_pandas(gdf_known)

    uniq_samples = df["sampleID"].unique()

    df_known = df_known[df_known["sampleID"].isin(uniq_samples)]
    dfk = None if d == "unknown" or df_known.empty else df_known.copy()
    if dfk is not None:
        dfk = dfk.dropna(subset=["x", "y"], how="any")
        dfk = None if dfk.empty else dfk.sort_values(by="sampleID")
    return dfk


def main(args):

    # Define and create output directory
    outdir = Path(args.output_dir)
    Path(outdir, "plots", "shapefile").mkdir(exist_ok=True, parents=True)

    gray_counties = "Benton,Washington,Scott,Crawford,Washington,Sebastian,Yell,Logan,Franklin,Madison,Carroll,Boone,Newton,Johnson,Pope,Van Buren,Searcy,Marion,Baxter,Stone,Independence,Jackson,Randolph,Bradley,Union,Ashley"

    coords_df = read_csv_with_dynamic_sep(args.coords_file)

    gg_dir = Path(args.geogenie_dir) / "bootstrap_predictions" / "test"
    dfgg = read_and_combine_csv_data([gg_dir], suffix="*_test_predictions.csv")

    loc_dir = Path(args.locator_dir) / "test"
    dfloc = read_and_combine_csv_data([loc_dir], suffix="*_test_predlocs.txt")
    dfgg["dataset"] = "test"
    dfgg = dfgg.sort_values(by="sampleID")

    for software, df in zip(["geogenie", "locator"], [dfgg, dfloc]):
        dataset_dir = outdir / software / "test"
        shpdir = dataset_dir / "plots" / "shapefile"
        Path(shpdir).mkdir(exist_ok=True, parents=True)

        print(f"Saving {software} test set plots to: {dataset_dir}")

        plotting = PlotGenIE(
            device="cpu",
            output_dir=dataset_dir,
            prefix=f"{software}_test",
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

        df_known = coords_df.copy()
        df_known = df_known.dropna(axis=0, how="any")

        make_all_plots(plotting, df, "test", df_known=df_known)
        print(f"Done with plotting iteration for dataset: test.")


if __name__ == "__main__":
    main(parse_arguments())
