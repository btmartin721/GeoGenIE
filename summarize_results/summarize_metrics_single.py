import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

fontsize = 13
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


def read_json_files(directory):
    """
    Reads all JSON files in the specified directory that match the pattern '*_test_metrics.json'
    and aggregates their contents into a pandas DataFrame, focusing only on selected metrics.
    """
    data = []
    for f in os.listdir(directory):
        if f.endswith("_test_metrics.json"):
            with open(os.path.join(directory, f), "r") as f:
                json_data = json.load(f)
                json_data["config"] = str(f).split("/")[-1].split(".")[0]
                data.append(json_data)
    return pd.DataFrame(data)


def create_facet_grid(df):
    """
    Creates a facet grid of histograms for each selected metric in the DataFrame.
    """
    plt.figure(figsize=(32, 48))

    df.drop(
        [
            "rho_p",
            "spearman_pvalue_longitude",
            "spearman_pvalue_latitude",
            "spearman_pvalue_longitude",
            "spearman_pvalue_latitude",
            "pearson_pvalue_latitude",
            "pearson_pvalue_longitude",
        ],
        axis=1,
        inplace=True,
    )

    df = df.loc[
        :,
        df.columns.isin(
            [
                "mean_dist",
                "median_dist",
                "stdev_dist",
                "percent_within_20km",
                "percent_within_50km",
                "percent_within_75km",
                "percentile_25",
                "percentile_50",
                "percentile_75",
                "percentiles_75",
                "mad_haversine",
                "coeffecient_of_variation",
                "pearson_corr_longitude",
                "pearson_corr_latitude",
                "spearman_corr_longitude",
                "spearman_corr_latitude",
                "skewness",
                "config",
            ]
        ),
    ]

    # Melting the DataFrame for FacetGrid compatibility
    df_melted = df.melt(var_name="Metric", value_name="Value", id_vars=["config"])

    df_melted = update_metric_labels(df_melted)

    df_melted.sort_values(by=["Metric", "config"], ascending=False, inplace=True)

    df_melted["config"] = df_melted["config"].str.split("_").str[:-2].str.join("_")

    col_order = [
        "Mean Error",
        "Median Error",
        "Median Absolute Deviation",
        "StdDev of Error",
        "25th Percentile of Error",
        "50th Percentile of Error",
        "75th Percentile of Error",
        "Coefficient of Variation",
        "Skewness",
        "% Samples within 20 km",
        "% Samples within 50 km",
        "% Samples within 75 km",
        "$R^2$ (Longitude)",
        "$R^2$ (Latitude)",
        "Rho (Longitude)",
        "Rho (Latitude)",
    ]

    df_melted = df_melted[df_melted["Metric"].isin(col_order)]
    df_melted["Metric"] = pd.Categorical(
        df_melted["Metric"], categories=col_order, ordered=True
    )
    df_melted.sort_values("Metric", inplace=True)

    df_melted, labels = update_config_labels(df_melted)

    col_wrap = 4

    metrics_requiring_reverse_palette = [
        "$R^2$ (Longitude)",
        "$R^2$ (Latitude)",
        "Rho (Longitude)",
        "Rho (Latitude)",
        "Skewness",
        "% Samples within 20 km",
        "% Samples within 50 km",
        "% Samples within 75 km",
    ]

    # Predefined y-axis order
    y_axis_order = [
        "Locator",
        "GeoGenie Base (Unoptimized)",
        "GeoGenie Base",
        "GeoGenie + Loss",
        "GeoGenie + Sampler",
        "GeoGenie + Loss + Sampler",
        "GeoGenie Base + Interpolation",
        "GeoGenie Loss + Interpolation",
        "GeoGenie Sampler + Interpolation",
        "GeoGenie Loss + Sampler + Interpolation",
    ]

    def map_palette(data, metric, ax, y_axis_order, metrics_requiring_reverse_palette):
        """
        Maps the appropriate color palette to the data for a given metric.

        Args:
            data (pd.DataFrame): The DataFrame containing the data for the current metric.
            metric (str): The name of the metric.
            ax (matplotlib.axes.Axes): The Axes object to plot on.
            y_axis_order (list): The order of categories on the y-axis.
            metrics_requiring_reverse_palette (list): Metrics that require a reversed color palette.
        """
        is_reverse_metric = metric in metrics_requiring_reverse_palette

        min_val, max_val = data["Value"].min(), data["Value"].max()
        normalized_values = (
            (data["Value"] - min_val) / (max_val - min_val)
            if max_val != min_val
            else np.zeros(len(data["Value"]))
        )

        # Choose the appropriate palette
        palette = sns.color_palette(
            "viridis" if is_reverse_metric else "viridis_r",
            as_cmap=True,
        )

        # Map normalized values to colors in the palette
        unique_configs = data["config"].unique()
        color_map = {
            config: palette(norm_val)
            for config, norm_val in zip(unique_configs, normalized_values)
        }

        # Create a barplot with the mapped colors
        sns.barplot(
            x="Value",
            y="config",
            hue="config",
            data=data,
            ax=ax,
            palette=color_map,
            order=y_axis_order,
        )

    # Initialize the FacetGrid object
    g = sns.FacetGrid(
        data=df_melted,
        col="Metric",
        col_wrap=col_wrap,
        sharex=False,
        col_order=col_order,
    )

    # Iterate over each subplot and apply the custom map_palette function
    for ax, (metric, metric_data) in zip(
        g.axes.flat, df_melted.groupby("Metric", observed=False)
    ):
        map_palette(
            metric_data, metric, ax, y_axis_order, metrics_requiring_reverse_palette
        )
        ax.set_title(metric, fontsize=17)
        ax.set_ylabel("Configuration", fontsize=17)
        ax.set_xlabel("Metric Value", fontsize=17)
        ax.tick_params(axis="both", which="major", labelsize=15)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save the plot
    plt.savefig(
        "all_comparisons_final/summary_facet_grid_selected_metrics.png",
        facecolor="white",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print(
        "Facet grid plot saved to all_comparisons_final/summary_facet_grid_selected_metrics.png."
    )


def update_metric_labels(df):
    """
    Update metric labels in the dataframe based on specified mappings.

    Args:
        df (pd.DataFrame): The dataframe to be updated.

    Returns:
        pd.DataFrame: The updated dataframe.
    """
    metric_map = {
        "mean_dist": "Mean Error",
        "median_dist": "Median Error",
        "stdev_dist": "StdDev of Error",
        "percent_within_20km": "% Samples within 20 km",
        "percent_within_50km": "% Samples within 50 km",
        "percent_within_75km": "% Samples within 75 km",
        "mad_haversine": "Median Absolute Deviation",
        "coeffecient_of_variation": "Coefficient of Variation",
        "percentile_25": "25th Percentile of Error",
        "percentile_50": "50th Percentile of Error",
        "percentiles_75": "75th Percentile of Error",  # typo in the original data
        "pearson_corr_longitude": "$R^2$ (Longitude)",
        "pearson_corr_latitude": "$R^2$ (Latitude)",
        "spearman_corr_longitude": "Rho (Longitude)",
        "spearman_corr_latitude": "Rho (Latitude)",
        "skewness": "Skewness",
    }

    for original, new in metric_map.items():
        df.loc[df["Metric"] == original, "Metric"] = new

    return df


def update_config_labels(df):
    """
    Update config labels in the dataframe based on the file list patterns.

    Args:
        df (pd.DataFrame): The dataframe to be updated.

    Returns:
        pd.DataFrame: The updated dataframe.
    """
    # Mapping of file name starts to corresponding labels
    config_map = {
        "locator": "Locator",
        "nn_base_unopt": "GeoGenie Base (Unoptimized)",
        "nn_base_opt": "GeoGenie Base",
        "nn_loss_opt": "GeoGenie + Loss",
        "nn_sampler_opt": "GeoGenie + Sampler",
        "nn_both_opt": "GeoGenie + Loss + Sampler",
        "nn_base_smote_opt": "GeoGenie Base + Interpolation",
        "nn_loss_smote_opt": "GeoGenie Loss + Interpolation",
        "nn_sampler_smote_opt": "GeoGenie Sampler + Interpolation",
        "nn_both_smote_opt": "GeoGenie Loss + Sampler + Interpolation",
    }
    # Iterate over the mapping and update the dataframe
    for key, value in config_map.items():
        df.loc[df["config"].str.startswith(key), "config"] = value

    return df, list(config_map.values())


# Example usage
directory_path = "all_comparisons_final/"
df = read_json_files(directory_path)
create_facet_grid(df)
