import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns

fontsize = 14
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
    plt.figure(figsize=(32, 32))

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
                "mad_haversine",
                "coeffecient_of_variation",
                "config",
            ]
        ),
    ]

    # Melting the DataFrame for FacetGrid compatibility
    df_melted = df.melt(var_name="Metric", value_name="Value", id_vars=["config"])

    df_melted.loc[df_melted["Metric"] == "mean_dist", "Metric"] = "Mean Error"

    df_melted.loc[df_melted["Metric"] == "median_dist", "Metric"] = "Median Error"
    df_melted.loc[df_melted["Metric"] == "stdev_dist", "Metric"] = "StdDev of Error"
    df_melted.loc[
        df_melted["Metric"] == "percent_within_20km", "Metric"
    ] = "Percent within 20 km"
    df_melted.loc[
        df_melted["Metric"] == "percent_within_50km", "Metric"
    ] = "Percent within 50 km"
    df_melted.loc[
        df_melted["Metric"] == "percent_within_75km", "Metric"
    ] = "Percent within 75 km"

    df_melted["config"] = df_melted["config"].str.replace("nn_", "dl_")

    df_melted.sort_values(by=["Metric", "config"], ascending=False, inplace=True)

    df_melted["config"] = df_melted["config"].str.split("_").str[:-2].str.join("_")

    df_melted.loc[df_melted["config"].str.startswith("refactor"), "config"] = "GeoGenie"
    df_melted.loc[df_melted["config"].str.startswith("original"), "config"] = "Locator"

    col_wrap = 4
    g = sns.FacetGrid(
        data=df_melted,
        col="Metric",
        col_wrap=col_wrap,
        hue="config",
        palette="Set2",
        sharey=False,
    )
    g.map(sns.barplot, "config", "Value")
    plt.subplots_adjust(hspace=1.5, wspace=1.5)

    # Total number of plots
    total_plots = len(df_melted["Metric"].unique())

    # Calculate the starting index of the bottom row
    start_of_bottom_row = total_plots - (total_plots % col_wrap or col_wrap)

    # Iterate through all axes
    for i, ax in enumerate(g.axes):
        # Clear the x-label for all axes except for those in the bottom row
        if i < start_of_bottom_row:
            ax.tick_params(labelbottom=False)

    # Adjusting plot aesthetics
    plt.tight_layout()

    # Save the plot
    plt.savefig("final_analysis/summary_facet_grid_selected_metrics.png")
    plt.close()
    print(
        "Facet grid plot saved to final_analysis/plots/summary_facet_grid_selected_metrics.pdf."
    )


# Example usage
directory_path = "final_analysis/"
df = read_json_files(directory_path)
create_facet_grid(df)
