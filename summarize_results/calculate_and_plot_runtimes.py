import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

fontsize = 20
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


def map_config_to_label(config):
    """
    Map the given config string to a more descriptive label.
    """
    parts = config.split("_")

    if len(parts) < 6:
        config_type = parts[0]
        out_value = ""
        meth_value = ""
    else:
        try:
            config_type = parts[13]
            out_value = parts[25]
            meth_value = parts[3]
        except IndexError:
            raise ValueError(f"Unexpected format in config: {config}")

    translocations_label = "Outliers Removed" if out_value == "true" else ""
    meth_label = "Oversampling" if meth_value == "kmeans" else ""

    if config_type == "loss":
        weight_label = "Sample Weighting"
    elif config_type == "none":
        weight_label = "No Sample Weights"
    elif config_type == "original" or config_type == "locator":
        weight_label = "Original Locator"
    else:
        raise ValueError(f"Unexpected configuration type: {config_type}")

    label = (
        f"{weight_label} + {translocations_label}"
        if translocations_label != ""
        else f"{weight_label}"
    )
    return f"{label} + {meth_label}" if meth_label != "" else f"{label}"


def read_and_combine_csv_data(directories):
    """
    Read CSV files from given directories and combine them into a single DataFrame.

    Args:
        directories (list of str): List of directory names to read the CSV files from.

    Returns:
        pd.DataFrame: Combined DataFrame with all the runtime data.
    """
    all_data = pd.DataFrame()
    for directory in directories:
        for csv_path in Path(directory).glob("*_execution_times.csv"):
            temp_df = pd.read_csv(csv_path)
            temp_df["Configuration"] = csv_path.stem  # File stem as configuration
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
    return all_data


# List of directories to search for CSV files
directories = [
    "/Users/btm002/Documents/wtd/GeoGenIE/analyses/all_model_outputs_final_really_really5/benchmarking",
    "/Users/btm002/Documents/wtd/GeoGenIE/analyses/original_locator/locator/benchmarking",
]

# Load and combine data from CSV files
combined_data = read_and_combine_csv_data(directories)

# Apply configuration mapping
combined_data["config"] = combined_data["Configuration"].apply(map_config_to_label)

# Add 'Model' column based on configurations for plotting
combined_data["Model"] = combined_data["config"]

combined_data["Program"] = "Original Locator"
combined_data.loc[combined_data["Model"] != "Original Locator", "Program"] = "GeoGenIE"


hue_order = [
    "Original Locator",
    "No Sample Weights",
    "Sample Weighting",
    "No Sample Weights + Outliers Removed",
    "Sample Weighting + Outliers Removed",
    "No Sample Weights + Oversampling",
    "Sample Weighting + Oversampling",
    "No Sample Weights + Outliers Removed + Oversampling",
    "Sample Weighting + Outliers Removed + Oversampling",
]

combined_data["Program"] = pd.Categorical(
    combined_data["Program"], categories=["Original Locator", "GeoGenIE"], ordered=True
)

combined_data["Model"] = pd.Categorical(
    combined_data["Model"], categories=hue_order, ordered=True
)

fig, ax = plt.subplots(1, 1, figsize=(16, 12))

# Hide the right and top spines
ax.spines[["right", "top"]].set_visible(False)

# Create boxplot
ax = sns.boxplot(
    x="Program",
    y="Execution Time",
    data=combined_data,
    hue="Model",
    hue_order=hue_order,
    palette="Set1",
)

ax.legend(title="Model Configuration", loc="upper left")
sns.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2)
plt.xlabel("Software Package")
plt.ylabel("Runtime (seconds)")


fig.savefig(
    "/Users/btm002/Documents/wtd/GeoGenIE/summarize_results/plots/runtimes_models.pdf",
    facecolor="white",
    bbox_inches="tight",
)
