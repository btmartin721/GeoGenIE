import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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
    Reads all JSON files in the specified directory that match the pattern '*_metrics.json'
    and aggregates their contents into a pandas DataFrame, focusing only on selected metrics.
    """
    data = []
    for f in os.listdir(directory):
        if f.endswith("_metrics.json"):
            with open(os.path.join(directory, f), "r") as f:
                json_data = json.load(f)
                json_data["config"] = str(f).split("/")[-1].split(".")[0]
                data.append(json_data)
    return pd.DataFrame(data)


def create_facet_grid(df):
    """
    Creates a facet grid of histograms for each selected metric in the DataFrame.
    """
    fig, ax = plt.subplots(2, 1, figsize=(16, 12))

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
                "pearson_corr_longitude",
                "pearson_corr_latitude",
                "config",
            ]
        ),
    ]

    df1 = df[["mean_dist", "median_dist", "stdev_dist", "config"]]
    df2 = df[["pearson_corr_longitude", "pearson_corr_latitude", "config"]]

    # Melting the DataFrame for FacetGrid compatibility
    df_melted1 = df1.melt(var_name="Metric", value_name="Value", id_vars=["config"])
    df_melted2 = df2.melt(var_name="Metric", value_name="Value", id_vars=["config"])

    df_melted1 = update_metric_labels(df_melted1)
    df_melted2 = update_metric_labels(df_melted2)

    df_melted1.sort_values(by=["Metric", "config"], ascending=False, inplace=True)
    df_melted2.sort_values(by=["Metric", "config"], ascending=False, inplace=True)

    df_melted1["config"] = df_melted1["config"].str.split("_").str[:-2].str.join("_")
    df_melted2["config"] = df_melted2["config"].str.split("_").str[:-2].str.join("_")

    col_order1 = ["Mean Error", "Median Error", "StdDev of Error"]
    col_order2 = ["$R^2$ (Longitude)", "$R^2$ (Latitude)"]

    df_melted1 = df_melted1[df_melted1["Metric"].isin(col_order1)]
    df_melted1["Metric"] = pd.Categorical(
        df_melted1["Metric"], categories=col_order1, ordered=True
    )
    df_melted1.sort_values("Metric", inplace=True)

    df_melted1, labels1 = update_config_labels(df_melted1)

    df_melted2 = df_melted2[df_melted2["Metric"].isin(col_order2)]
    df_melted2["Metric"] = pd.Categorical(
        df_melted2["Metric"], categories=col_order2, ordered=True
    )
    df_melted2.sort_values("Metric", inplace=True)

    df_melted2, labels2 = update_config_labels(df_melted2)

    # Initialize the FacetGrid object
    g1 = sns.boxplot(
        data=df_melted1,
        x="Metric",
        y="Value",
        hue="config",
        hue_order=labels1,
        ax=ax[0],
    )
    g2 = sns.boxplot(
        data=df_melted2,
        x="Metric",
        y="Value",
        hue="config",
        hue_order=labels2,
        ax=ax[1],
    )

    sns.move_legend(ax[0], loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=17)
    sns.move_legend(ax[1], loc="upper left", bbox_to_anchor=(1.04, 1.0), fontsize=17)

    ax[0].set_xlabel("Metric", fontsize=24)
    ax[1].set_xlabel("Metric", fontsize=24)
    ax[0].set_ylabel("Value", fontsize=24)
    ax[1].set_ylabel("Value", fontsize=24)
    ax[1].set_ylim([0, 1])

    # fig.subplots_adjust(hspace=0.55)

    # plt.subplots_adjust(hspace=0.5)

    # # Iterate over each subplot and apply the custom map_palette function
    # for ax, (metric, metric_data) in zip(
    #     g.axes.flat, df_melted.groupby("Metric", observed=False)
    # ):
    #     map_palette(
    #         metric_data, metric, ax, y_axis_order, metrics_requiring_reverse_palette
    #     )
    #     ax.set_title(metric, fontsize=17)
    #     ax.set_ylabel("Configuration", fontsize=17)
    #     ax.set_xlabel("Metric Value", fontsize=17)
    #     ax.tick_params(axis="both", which="major", labelsize=14)

    # plt.subplots_adjust(hspace=0.5, wspace=0.5)

    # Save the plot
    fig.savefig(
        "final_analysis_boot/summary_facet_grid_selected_metrics.png",
        facecolor="white",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print(
        "Facet grid plot saved to final_analysis_boot/summary_facet_grid_selected_metrics.png."
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
        "original": "Locator",
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
directory_path = "final_analysis_boot/"
df = read_json_files(directory_path)
create_facet_grid(df)
