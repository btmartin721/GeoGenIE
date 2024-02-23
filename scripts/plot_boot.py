import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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

plt.rcParams.update(sizes)
mpl.rcParams.update(sizes)


def load_data(directory):
    """
    Load all CSV files from the given directory and concatenate them into a single GeoDataFrame.

    Args:
        directory (str): Path to the directory containing CSV files.

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame containing all data.
    """
    files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".csv") or f.endswith(f"txt")
    ]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
    return gdf


def calculate_centroid(gdf):
    """
    Calculate the centroid of a set of geographic points.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.

    Returns:
        gpd.GeoSeries: Centroids.
    """
    return gdf.geometry.unary_union.centroid


def extract_shapefile(zip_path):
    """
    Extract a zipped shapefile.

    Args:
        zip_path (str): Path to the zipped shapefile.

    Returns:
        str: Path to the extracted shapefile directory.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


def plot_sample(
    sample_gdf,
    sample_id,
    county_shapefile_dir,
    output_dir,
    prefix,
    gray_counties=None,
    sample_gdf_known=None,
):
    """
    Plot the bootstrapped points using a Gaussian KDE, including contour lines for density, and highlight the centroid for a given sample on a basemap with US state county lines.

    Args:
        sample_gdf (gpd.GeoDataFrame): GeoDataFrame containing coordinates for a single sample.
        sample_id (str): Sample ID to plot.
        county_shapefile_dir (str): Directory containing the county shapefile.
        output_dir (str): Directory to save the figures.
        prefix (str): Prefix for the figure filenames.
        gray_counties (list): List of counties to color gray. If None, then doesn't color any counties gray. Defaults to None.
        sample_gdf_known (GeoPandas.DataFrame): GeoPandas DataFrame with known coordinates for predicted values. Defaults to None.
    """
    coords = np.array(list(sample_gdf.geometry.apply(lambda p: (p.x, p.y))))

    # Calculate Mean Coordinates
    centroids = calculate_centroid(sample_gdf)

    if sample_gdf_known is not None:
        centroid_known = calculate_centroid(sample_gdf_known)

    mean_lon, mean_lat = centroids.x, centroids.y

    # Calculate bounds with buffer
    xmin, ymin, xmax, ymax = sample_gdf.total_bounds

    # Load county shapefile as basemap
    county_gdf = gpd.read_file(county_shapefile_dir)
    county_gdf = county_gdf.to_crs("EPSG:4326")

    # Filter for Arkansas counties if the shapefile contains all US counties
    arkansas_counties = county_gdf[
        county_gdf["STATEFP"] == "05"
    ]  # FIPS code for Arkansas

    fig, ax = plt.subplots(figsize=(8, 6))

    if gray_counties is not None:
        # Filtering the counties to be colored gray
        gray_county_gdf = arkansas_counties[
            arkansas_counties["NAME"].isin(gray_counties)
        ]
        gray_county_gdf.plot(ax=ax, color="darkgray", edgecolor="k", alpha=0.5)

    # Plot all counties, then overlay the specified counties in gray
    arkansas_counties.plot(ax=ax, color="none", edgecolor="black")

    # Apply Gaussian Kernel Density Estimation
    kde = gaussian_kde(coords.T)

    # Create grid for KDE
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    z = kde(grid)
    zz = np.reshape(z, xx.shape)

    # Calculate cumulative density for the levels
    z_sorted = np.sort(z.ravel())
    cumulative_z = np.cumsum(z_sorted)
    total_z = cumulative_z[-1]

    # Find levels for 50%, 70%, and 90% of the total cumulative density
    level_50 = z_sorted[np.searchsorted(cumulative_z, 0.50 * total_z)]
    level_70 = z_sorted[np.searchsorted(cumulative_z, 0.70 * total_z)]
    level_90 = z_sorted[np.searchsorted(cumulative_z, 0.90 * total_z)]

    # Ensure levels are in ascending order
    levels = np.sort([level_50, level_70, level_90])

    # Plot density contours as lines
    # Use a colorblind-friendly palette
    colors = ["#E69F00", "#56B4E9", "#CC79A7"]  # Orange, Sky Blue, Magenta

    # Plot density contours as lines
    contour = ax.contour(xx, yy, zz, levels=levels, colors=colors)

    # Create custom legend for the contours
    contour_lines = [mlines.Line2D([0], [0], color=color, lw=2) for color in colors]

    grays = [mpatches.Patch(facecolor="darkgrey", edgecolor="k")]

    labels = ["90% Density", "70% Density", "50% Density"]

    # Plot bootstrapped points with reduced opacity
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.3, color="gray")

    # Ensure the mean marker is visible
    ax.scatter(
        mean_lon, mean_lat, s=100, color="k", marker="X", label="Predicted Locality"
    )

    if sample_gdf_known is not None:
        ax.scatter(
            centroid_known.x,
            centroid_known.y,
            s=100,
            color="k",
            marker="^",
            label="Recorded Locality",
        )

    handles, labs = ax.get_legend_handles_labels()
    contour_lines += handles + grays
    labels += labs + ["CWD Mgmt Zone"]

    ax.legend(
        contour_lines,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 1.3),
        fancybox=True,
        shadow=True,
        ncol=2,
    )
    ax.set_title(f"Sample {sample_id}")

    # Save figure
    plt.savefig(
        os.path.join(output_dir, f"{prefix}_{sample_id}.png"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main(
    directory,
    basemap_path,
    output_dir,
    prefix,
    selected_samples=None,
    gray_counties=None,
    known_coords=None,
):
    """
    Main function to process and plot data for selected samples.

    Args:
        directory (str): Directory containing CSV files.
        basemap_path (str): Path to the zipped basemap shapefile.
        output_dir (str): Directory to save the plots.
        prefix (str): Prefix for plot filenames.
        selected_samples (list of str, optional): List of sampleIDs to be plotted. Plots all if None.
        buffer (float): Buffer to add around the sample bounding box. Defaults to 0.1.
        gray_counties (list): List of counties to color gray. Defaults to None.
        known_coords (str): Path to known coordinates for predicted samples. For validation. If None, then doesn't plot the real coordinates. Defaults to None.
    """
    gdf = load_data(directory)
    df_known = pd.read_csv(known_coords, sep="\t")

    gdf_known = None
    if known_coords is not None:
        gdf_known = gpd.GeoDataFrame(
            df_known, geometry=gpd.points_from_xy(df_known.x, df_known.y)
        )

    basemap_dir = extract_shapefile(basemap_path)

    Path(output_dir).mkdir(exist_ok=True, parents=True)

    for sample_id in gdf["sampleID"].unique():
        if selected_samples is None or sample_id in selected_samples:
            sample_gdf = gdf[gdf["sampleID"] == sample_id]
            sample_gdf_known = gdf_known[gdf_known["sampleID"] == sample_id]
            plot_sample(
                sample_gdf,
                sample_id,
                basemap_dir,
                output_dir,
                prefix,
                gray_counties=gray_counties,
                sample_gdf_known=sample_gdf_known,
            )


mz = [
    "Benton",
    "Wasnington",
    "Scott",
    "Crawford",
    "Washington",
    "Sebastian",
    "Yell",
    "Logan",
    "Franklin",
    "Madison",
    "Carroll",
    "Boone",
    "Newton",
    "Johnson",
    "Pope",
    "Van Buren",
    "Searcy",
    "Marion",
    "Baxter",
    "Stone",
    "Independence",
    "Jackson",
    "Randolph",
    "Bradley",
    "Union",
    "Ashley",
]
directory = "boot_fy23_locator"
basemap_path = "data/cb_2018_us_county_5m.zip"
output_dir = "final_boot_results_locator"
known_coords = "/Users/btm002/Documents/wtd/GeoGenIE/data/wtd_fy23_samples_known.txt"
prefix = "locator_fy23"
selected_samples = [
    "83RA6P16",
    "83RA6P27",
    "83IN6N47",
    "83IN6N62",
    "83CV6N21",
    "83CG6N17",
    "83CG6N18",
    "83PI6N14",
    "83PI6N15",
    "83PO6N18",
    "83HO6N21",
    "83YE6N57",
    "83DR6N23",
    "83PU6N25",
    "83MS6N6",
    "83MS6N7",
]
main(
    directory,
    basemap_path,
    output_dir,
    prefix,
    selected_samples=selected_samples,
    gray_counties=mz,
    known_coords=known_coords,
)
