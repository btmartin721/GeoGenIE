import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_and_plot_params(directory):
    """
    Reads JSON files from a given directory and plots parameter distributions.

    Args:
        directory (str): Path to the directory containing JSON files.

    Returns:
        None: Plots are displayed but not returned.
    """

    # Step 1: Read JSON files and aggregate data
    params_list = []
    for file in os.listdir(directory):
        if file.endswith("_best_params.json"):
            with open(os.path.join(directory, file), "r") as f:
                data = json.load(f)

                # Default missing boolean values to False
                if "use_weighted" not in data:
                    data["use_weighted"] = False

                params_list.append(data)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(params_list)

    print(df)

    # Step 2: Plotting
    # Contour Plot
    g = sns.PairGrid(df)
    g.map_upper(sns.kdeplot, levels=4, color=".2", warn_singular=False)
    g.map_lower(sns.kdeplot, cmap="viridis", warn_singular=False)
    g.map_diag(sns.kdeplot, lw=2, warn_singular=False)
    plt.show()

    # Identify continuous and categorical columns
    continuous_cols = df.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = df.select_dtypes(include=["bool", "object"]).columns

    # Melt the DataFrame for FacetGrid compatibility
    df_melted_continuous = df.melt(value_vars=continuous_cols)
    df_melted_categorical = df.melt(value_vars=categorical_cols)

    # Step 2: Plotting for continuous columns
    g_cont = sns.FacetGrid(
        df_melted_continuous,
        col="variable",
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=3,
    )
    g_cont.map(sns.kdeplot, "value", fill=True, color="darkorchid")
    g_cont.set_titles("{col_name}")
    plt.show()

    # Step 3: Plotting for categorical columns
    g_cat = sns.FacetGrid(
        df_melted_categorical,
        col="variable",
        col_wrap=4,
        sharex=False,
        sharey=False,
        height=3,
    )

    g_cat.map(sns.countplot, "value", color="darkorchid", alpha=0.6)
    g_cat.set_titles("{col_name}")
    plt.show()


# Example usage
read_and_plot_params("final_analysis/optimize/")
