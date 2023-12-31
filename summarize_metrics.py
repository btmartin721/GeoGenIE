import os
import sys
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
                json_data["config"] = "_".join(str(f).split("/")[-1].split("_")[2:-2])
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
        ],
        axis=1,
        inplace=True,
    )

    # Melting the DataFrame for FacetGrid compatibility
    df_melted = df.melt(var_name="Metric", value_name="Value")

    # Creating FacetGrid
    g = sns.FacetGrid(df_melted, col="Metric", col_wrap=6, sharex=False, sharey=False)
    g.map(sns.histplot, "Value", kde=True)

    # Adjusting plot aesthetics
    for ax in g.axes:
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    plt.tight_layout()

    # Save the plot
    plt.savefig("final_analysis/plots/summary_facet_grid_selected_metrics.png")
    plt.close()
    print(
        "Facet grid plot saved to final_analysis/plots/summary_facet_grid_selected_metrics.png."
    )


# Example usage
directory_path = "final_analysis/test"
df = read_json_files(directory_path)
create_facet_grid(df)
