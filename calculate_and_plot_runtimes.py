import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_and_combine_data(directories):
    """
    Read CSV files from given directories and combine them into a single DataFrame.

    Args:
        directories (list of str): List of directory names to read the CSV files from.

    Returns:
        pandas.DataFrame: Combined DataFrame with all the runtime data.
    """
    all_data = pd.DataFrame()

    for directory in directories:
        csv_path = os.path.join(
            directory, "benchmarking", f"{directory}_execution_times.csv"
        )
        if os.path.exists(csv_path):
            temp_df = pd.read_csv(csv_path)
            temp_df["Directory"] = directory
            all_data = pd.concat([all_data, temp_df], ignore_index=True)
        else:
            print(f"File not found: {csv_path}")

    return all_data


# List of directories
directories = [
    "gb_none_none2",
    "gb_loss_none2",
    "gb_loss_smote2",
    "gb_none_smote2",
    "nn_none_none",
    "nn_loss_none",
    "nn_sampler_none",
    "nn_both_none",
    "nn_none_smote",
    "nn_loss_smote",
    "nn_sampler_smote",
    "nn_both_smote",
    "locator",
]

# Combine data from all directories
combined_data = read_and_combine_data(directories)

# # Create boxplot
# sns.boxplot(x="Directory", y="Execution Time", data=combined_data)
# plt.xticks(rotation=45)
# plt.title("Runtime Comparison Across Model Configurations")
# plt.xlabel("Configuration")
# plt.ylabel("Runtime (seconds)")
# plt.tight_layout()
# plt.savefig("runtimes.png", facecolor="white")
# plt.show()

combined_data["Model"] = combined_data["Function Name"]
combined_data.loc[
    combined_data["Function Name"] == "train_rf",
    "Model",
] = "Gradient Boosting"
combined_data.loc[
    combined_data["Function Name"] == "train_model",
    "Model",
] = "Deep Learning"
combined_data.loc[
    combined_data["Function Name"] == "locator", "Model"
] = "Original Locator"

combined_data["Model_cat"] = pd.Categorical(
    combined_data["Model"],
    categories=["Original Locator", "Gradient Boosting", "Deep Learning"],
    ordered=True,
)

combined_data.loc[
    combined_data["Directory"].str.contains("none_none"), "Configuration"
] = "Base Model"
combined_data.loc[
    combined_data["Directory"].str.contains("loss_none"), "Configuration"
] = "Sample Weighting (Loss)"
combined_data.loc[
    combined_data["Directory"].str.contains("sampler_none"), "Configuration"
] = "Sample Weighting (Sampler)"
combined_data.loc[
    combined_data["Directory"].str.contains("none_smote"), "Configuration"
] = "Over-sampling"
combined_data.loc[
    combined_data["Directory"].str.contains("loss_smote"), "Configuration"
] = "Weighting (Loss) + Over-sampling"
combined_data.loc[
    combined_data["Directory"].str.contains("sampler_smote"), "Configuration"
] = "Weighting (Sampler) + Over-sampling"

combined_data.loc[
    combined_data["Model_cat"] == "Original Locator", "Configuration"
] = "Base Model"


pal = sns.color_palette("tab10", n_colors=6)

# Create boxplot
sns.boxplot(
    x="Model_cat",
    y="Execution Time",
    data=combined_data,
    hue="Configuration",
    hue_order=[
        "Base Model",
        "Sample Weighting (Loss)",
        "Sample Weighting (Sampler)",
        "Over-sampling",
        "Weighting (Loss) + Over-sampling",
        "Weighting (Sampler) + Over-sampling",
    ],
    palette=pal,
)
# plt.xticks(rotation=90)
plt.title("Runtime Comparison Across Models")
plt.xlabel("Model Configuration")
plt.ylabel("Runtime (seconds)")
plt.legend(loc="best", fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig("runtimes_models.pdf", facecolor="white")
plt.show()
