import json
import logging
import os
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import torch
import wget
from esda.moran import Moran
from libpysal.weights import DistanceBand
from optuna import exceptions as optuna_exceptions
from optuna import visualization
from scipy.spatial.distance import euclidean
from scipy.stats import gamma
from shapely.geometry import Point
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from geogenie.samplers.samplers import GeographicDensitySampler, custom_gpr_optimizer
from geogenie.utils.exceptions import TimeoutException
from geogenie.utils.scorers import haversine_distances_agg
from geogenie.utils.utils import load_directory_of_locs, time_limit

warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings(action="ignore", category=optuna_exceptions.ExperimentalWarning)


class PlotGenIE:
    def __init__(
        self,
        device,
        output_dir,
        prefix,
        basemap_fips,
        basemap_highlights,
        show_plots=False,
        fontsize=18,
        filetype="png",
        dpi=300,
    ):
        """
        A class dedicated to generating and managing plots for data visualization, particularly in the context of geographical and statistical data analysis. It provides functionalities to customize plot appearance and supports saving plots to specified output directories.

        Args:
            device (str): The device used for plotting, typically 'cpu' or 'cuda'.
            output_dir (str): The directory where plots will be saved.
            prefix (str): A prefix added to the names of saved plot files.
            basemap_fips (str): FIPS code for base map.
            basemap_highlights (list): List of counties to highlight gray on base map.
            show_plots (bool, optional): Flag to determine if plots should be displayed inline. Defaults to False.
            fontsize (int, optional): Font size used in plots. Defaults to 18.
            filetype (str, optional): File type/format for saving plots. Defaults to 'png'.
            dpi (int, optional): Dots per inch, specifying the resolution of plots. Defaults to 300.

        Attributes:
            device (str): The device used for plotting, typically 'cpu' or 'cuda'.
            output_dir (str): The directory where plots will be saved.
            prefix (str): A prefix added to the names of saved plot files.
            basemap_fips (str): FIPS code for base map.
            basemap_highlights (list): List of counties to highlight gray on base map.
            show_plots (bool): Flag to determine if plots should be displayed inline. Defaults to False.
            fontsize (int): Font size used in plots. Defaults to 18.
            filetype (str): File type/format for saving plots. Defaults to 'png'.
            dpi (int): Dots per inch, specifying the resolution of plots. Defaults to 300.
            outbasepath (Path): Base path for saving plots, constructed from output_dir, prefix, and filetype.
            logger (Logger): Logger for logging information and errors.

        Notes:
            - This class is designed to work with matplotlib and seaborn for generating plots.
            - Global matplotlib settings are adjusted according to the specified fontsize and dpi.
        """
        self.device = device
        self.output_dir = output_dir
        self.prefix = prefix
        self.basemap_fips = basemap_fips
        self.basemap_highlights = basemap_highlights
        self.show_plots = show_plots
        self.fontsize = fontsize
        self.filetype = filetype
        self.dpi = dpi

        self.outbasepath = Path(
            self.output_dir, "plots", f"{self.prefix}.{self.filetype}"
        )

        self.logger = logging.getLogger(__name__)

        # Adjust matplotlib settings globally.
        sizes = {
            "axes.labelsize": self.fontsize,
            "axes.titlesize": self.fontsize,
            "figure.titlesize": self.fontsize,
            "figure.labelsize": self.fontsize,
            "xtick.labelsize": self.fontsize,
            "ytick.labelsize": self.fontsize,
            "font.size": self.fontsize,
            "legend.fontsize": self.fontsize,
            "legend.title_fontsize": self.fontsize,
            "figure.dpi": self.dpi,
            "savefig.dpi": self.dpi,
        }

        sns.set_context("paper", rc=sizes)
        plt.rcParams.update(sizes)
        mpl.rcParams.update(sizes)

    def plot_times(self, rolling_avgs, rolling_stds, filename):
        """Plot model training times."""
        plt.figure(figsize=(10, 5))
        plt.plot(rolling_avgs, label="Rolling Average (Bootstrap Time)")
        plt.fill_between(
            range(len(rolling_avgs)),
            np.array(rolling_avgs) - np.array(rolling_stds),
            np.array(rolling_avgs) + np.array(rolling_stds),
            color="b",
            alpha=0.2,
            label="Standard Deviation (Bootstrap Time)",
        )
        plt.xlabel("Bootstrap Replicate")
        plt.ylabel("Duration (s)")
        plt.title(
            "Rolling Average Time of per-Bootstrap Model Training",
            fontsize=self.fontsize,
        )
        plt.legend()

        if self.show_plots:
            plt.show()
        plt.savefig(filename, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_smote_bins(
        self, df, bins, df_orig, bins_orig, url, buffer=0.1, marker_scale_factor=3
    ):
        """
        Plots scatter plots before and after SMOTE (Synthetic Minority Over-sampling Technique) to visualize the effect of oversampling on the data distribution. The method creates a subplot with two scatter plots: one showing the original data and the other showing the data after SMOTE has been applied.

        Args:
            df (pandas DataFrame): DataFrame containing the data after SMOTE oversampling.
            bins (array-like): Array of bin labels for the data after SMOTE.
            df_orig (pandas DataFrame): DataFrame containing the original data before SMOTE.
            bins_orig (array-like): Array of original bin labels before SMOTE.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.
            marker_scale_factor (int, optional): Factor to scale the size of markers in the scatter plot. Defaults to 3.

        Notes:
            - This function visually compares the geographical distribution of data before and after SMOTE.
            - Each plot shows data points colored by their bin labels, providing insight into the oversampling process.
        """
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))

        ax = self._plot_smote_scatter(
            df_orig, bins_orig, axs[0], "Before Simulations", "upper right", url, buffer
        )
        ax2 = self._plot_smote_scatter(
            df, bins, axs[1], "After Simulations", "upper left", url, buffer
        )

        plt.subplots_adjust(wspace=0.25)

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_oversampling_scatter.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def _plot_smote_scatter(self, df, bins, ax, title, loc, url, buffer):
        """
        Creates a scatter plot for visualizing data points with their associated bin labels.

        This method is used internally by `plot_smote_bins` to generate individual scatter plots.

        Args:
            df (pandas DataFrame): DataFrame containing the data to be plotted.
            bins (array-like): Array of bin labels for the data.
            ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
            title (str): Title of the scatter plot.
            loc (str): Location of the legend in the plot.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float): Buffer distance for geographical plotting.
            scale_factor (int): Factor to scale the size of markers in the scatter plot.

        Notes:
            - This function is a helper method and is not intended to be called directly.
            - It adds a scatter plot to the provided Axes object, with data points colored by their bin labels.
        """
        df = df.copy()
        df["bins"] = bins

        try:
            df["size"] = df["sample_weights"].astype(int)
        except (KeyError, TypeError):
            df["size"] = 1

        self._plot_map(df[["x", "y"]].to_numpy(), url, self.output_dir, buffer, ax)

        ax = sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="bins",
            size="size",
            sizes=(100, 1000),
            palette="Set2",
            ax=ax,
            alpha=0.7,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)

        anchor = (1.04, 1.0) if loc == "upper left" else (-0.2, 1.0)

        ax.legend(fontsize=self.fontsize, loc=loc, bbox_to_anchor=anchor)

    def plot_history(self, train_loss, val_loss):
        """Automatically plot training and validation loss with appropriate scaling.

        Args:
            train_loss (list): List of training loss values.
            val_loss (list): List of validation loss values.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.legend()

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_train_history.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def make_optuna_plots(self, study):
        """Visualize Optuna search using built-in Optuna plotting methods.

        Args:
            study (optuna.study): Optuna study to plot.
        """
        if not visualization.is_available():
            self.logger.warn(
                "Could not make plots because plotly and kaleido are not installed."
            )
            return

        try:
            importance_fig = visualization.plot_param_importances(study)
            importance_fig.write_image(
                self.outbasepath.with_name(
                    f"{self.prefix}_param_importances.{self.filetype}"
                )
            )

            edf_fig = visualization.plot_edf(study, target_name="Location Error")
            edf_fig.write_image(
                self.outbasepath.with_name(f"{self.prefix}_edf.{self.filetype}")
            )

            par_fig = visualization.plot_parallel_coordinate(
                study, target_name="Location Error"
            )

            par_fig.write_image(
                self.outbasepath.with_name(
                    f"{self.prefix}_parallel_coordinates.{self.filetype}"
                )
            )

            slice_fig = visualization.plot_slice(study, target_name="Location Error")
            slice_fig.write_image(
                self.outbasepath.with_name(f"{self.prefix}_slices.{self.filetype}")
            )

            tl_fig = visualization.plot_timeline(study)
            tl_fig.write_image(
                self.outbasepath.with_name(f"{self.prefix}_timeline.{self.filetype}")
            )

            rank_fig = visualization.plot_rank(study, target_name="Location Error")
            rank_fig.write_image(
                self.outbasepath.with_name(f"{self.prefix}_rank.{self.filetype}")
            )

            try:
                with time_limit(20):
                    ctr_fig = visualization.plot_contour(
                        study, target_name="Location Error"
                    )
                    ctr_fig.write_image(
                        self.outbasepath.with_name(
                            f"{self.prefix}_contour.{self.filetype}"
                        )
                    )
            except TimeoutException as e:
                self.logger.warning("Generation of Optuna contour plot timed out.")

            hist_fig = visualization.plot_optimization_history(
                study, target_name="Location Error"
            )
            hist_fig.write_image(
                self.outbasepath.with_name(f"{self.prefix}_opt_history.{self.filetype}")
            )

        except Exception as e:
            self.logger.error(f"Could not create plot: {e}")
            raise

    def _read_json_files(self, directory):
        """
        Reads all JSON files in the specified directory that match the pattern '*_test_metrics.json'
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

    def create_metrics_facet_grid(self):
        """
        Creates a facet grid of histograms for each selected metric in the DataFrame.
        """

        metric_dir = os.path.join(self.output_dir, "bootstrap")

        df = self._read_json_files(metric_dir)

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

        df_melted = self.update_metric_labels(df_melted)

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

        df_melted, labels = self.update_config_labels(df_melted)

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

        def map_palette(
            data, metric, ax, y_axis_order, metrics_requiring_reverse_palette
        ):
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

    def plot_bootstrap_aggregates(self, df):
        """Make KDE and bar plots with bootstrap distributions and CIs."""
        fig, ax = plt.subplots(2, 1, figsize=(8, 16))

        df_errordist = df[["mean_dist", "median_dist"]].copy()

        df_errordist.rename(
            columns={
                "mean_dist": "Mean Haversine Error (km)",
                "median_dist": "Median Haversine Error (km)",
            },
            inplace=True,
        )

        df_corr = df[
            [
                "pearson_corr_longitude",
                "pearson_corr_latitude",
                "spearman_corr_longitude",
                "spearman_corr_latitude",
            ]
        ].copy()

        df_corr.rename(
            columns={
                "pearson_corr_longitude": "$R^2$ (Longitude)",
                "pearson_corr_latitude": "$R^2$ (Latitude)",
                "spearman_corr_longitude": "Rho (Longitude)",
                "spearman_corr_latitude": "Rho (Latitude)",
            },
            inplace=True,
        )

        df_melt_errordist = df_errordist.melt()
        df_melt_corr = df_corr.melt()

        ax[0] = sns.kdeplot(
            data=df_melt_errordist,
            x="value",
            hue="variable",
            fill=True,
            palette="Set2",
            legend=True,
            ax=ax[0],
        )
        ax[1] = sns.kdeplot(
            data=df_melt_corr,
            x="value",
            hue="variable",
            fill=True,
            palette="Set2",
            legend=True,
            ax=ax[1],
        )

        ax[0].set_title("Distribution of Bootstrapped Error")
        ax[0].set_xlabel("Distance Error")
        ax[0].set_ylabel("Density")

        sns.move_legend(
            ax[0],
            loc="center",
            bbox_to_anchor=(0.5, 1.45),
            fancybox=True,
            shadow=True,
            ncol=len(df_melt_errordist["variable"].unique()),
        )

        ax[0].legend_.set_title("Metric")

        ax[1].set_title("Distribution of Bootstrapped Error")
        ax[1].set_xlabel("Correlation Coefficient")
        ax[1].set_ylabel("Density")
        sns.move_legend(
            ax[1],
            loc="center",
            bbox_to_anchor=(0.5, 1.45),
            fancybox=True,
            shadow=True,
            ncol=len(df_melt_corr["variable"].unique()) // 2,
        )

        ax[1].legend_.set_title("Metric")

        plt.subplots_adjust(hspace=1.05)

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_bootstrap_error_distributions.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

        pride = mcolors.LinearSegmentedColormap.from_list(
            "pride", ["#FF0018", "#FFA52C", "#FFFF41", "#008018", "#0000F9", "#86007D"]
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 12))
        ax[0] = sns.boxplot(
            data=df_melt_errordist,
            x="variable",
            y="value",
            hue="variable",
            fill=True,
            palette="Dark2",
            legend=True,
            ax=ax[0],
        )
        ax[1] = sns.boxplot(
            data=df_melt_corr,
            x="variable",
            y="value",
            hue="variable",
            palette="Dark2",
            legend=True,
            ax=ax[1],
        )

        ax[0].set_title("Bootstrapped Prediction Error")
        ax[0].set_xlabel("Metric")
        ax[0].set_ylabel("Prediction Error (km)")
        ax[0].set_xticklabels("")
        sns.move_legend(
            ax[0], loc="center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True
        )

        ax[1].set_title("Bootstrapped Prediction Error")
        ax[1].set_xlabel("Metric")
        ax[1].set_ylabel("Correlation Coefficient")
        sns.move_legend(
            ax[1],
            loc="center",
            bbox_to_anchor=(0.5, 1.15),
            fancybox=True,
            shadow=True,
            ncol=2,
        )
        ax[1].set_xticklabels("")

        plt.subplots_adjust(wspace=1.05)

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_bootstrap_error_barplots.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def update_metric_labels(self, df):
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

    def update_config_labels(self, df):
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

    def plot_scatter_samples_map(
        self,
        y_true_train,
        y_true,
        dataset,
        url,
        buffer=0.1,
    ):
        """
        Plots geographical scatter plots of training and test/validation sample densities. This method creates a subplot with two scatter plots, one showing the density of training samples and the other for test or validation samples.

        Args:
            y_true_train (np.array): Array of actual geographical coordinates for the training dataset.
            y_true (np.array): Array of actual geographical coordinates for the test or validation dataset.
            dataset (str): Specifies whether the dataset is 'test' or 'validation'.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.

        Notes:
            - The method visualizes the geographical distribution of training and test/validation samples.
            - It uses scatter plots to represent the density of samples in different geographical areas.
            - The scatter plots are overlaid on top of a base map obtained from the specified shapefile URL.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 12))

        # Plot KDE as contour on the maps
        for i, title, y in zip(
            range(2),
            ["Training Sample Density", f"{dataset.capitalize()} Sample Density"],
            [y_true_train, y_true],
        ):
            # Assuming self._plot_map is a method to plot the map on ax
            self._plot_map(y, url, self.output_dir, buffer, ax[i])

            ax[i] = sns.scatterplot(
                x=y[:, 0],
                y=y[:, 1],
                s=plt.rcParams["lines.markersize"] ** 2 * 4,
                c="darkorchid",
                alpha=0.6,
                ax=ax[i],
            )

            ax[i].set_title(title)
            ax[i].set_xlabel("Longitude")
            ax[i].set_ylabel("Latitude")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_train_{dataset}_sample_densities.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def _calculate_vector_errors(self, actual_coords, predicted_coords):
        """
        Calculates the vector difference errors between actual and predicted coordinates.
        """
        vector_errors = np.array(
            [euclidean(act, pred) for act, pred in zip(actual_coords, predicted_coords)]
        )
        return vector_errors

    def _calculate_spatial_autocorrelation(self, errors, coords):
        """
        Calculates spatial autocorrelation (Moran's I) for the errors.
        """
        weights = DistanceBand.from_array(
            coords, threshold=1.0
        )  # Adjust threshold based on your data
        mi = Moran(errors, weights)
        return mi.I

    def _run_gpr_with_matern_kernel(
        self, actual_coords, vector_errors, n_restarts_optimizer=50
    ):
        """
        Runs Gaussian Process Regression using the Matern kernel.
        """
        kernel = 1 * Matern(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e6)
        ) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-10, 1e6))
        gp = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=n_restarts_optimizer
        )
        gp.fit(actual_coords, vector_errors)
        return gp

    def plot_geographic_error_distribution(
        self,
        actual_coords,
        predicted_coords,
        url,
        dataset,
        buffer=0.1,
        marker_scale_factor=2,
        min_colorscale=0,
        max_colorscale=300,
        n_contour_levels=20,
        centroids=None,
    ):
        """
        Plots the geographic distribution of prediction errors and their uncertainties. This function calculates the Haversine error between actual and predicted coordinates and uses Gaussian Process Regression (GPR) to estimate error and uncertainty across a geographical area.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            predicted_coords (np.array): Array of predicted geographical coordinates.
            url (str): URL for the shapefile to plot geographical data.
            dataset (str): Name of the dataset being used.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.
            marker_scale_factor (int, optional): Scale factor for marker size in plots. Defaults to 3.
            min_colorscale (int, optional): Minimum value for the color scale. Defaults to 0.
            max_colorscale (int, optional): Maximum value for the color scale. Defaults to 300.
            n_contour_levels (int, optional): Number of contour levels in the plot. Defaults to 20.
            centroids (np.array, optional): Array of centroids to be plotted. Defaults to None.

        Notes:
            - This method produces two subplots: one showing the spatial distribution of prediction errors and the others showing the uncertainty of these predictions.
        """
        # # Calculate Haversine error for each pair of points
        # haversine_errors = haversine_distances_agg(
        #     actual_coords, predicted_coords, np.array
        # )

        # if len(haversine_errors.shape) > 1:
        #     msg = f"Invalid shape found in haversine_error estimations: {haversine_errors.shape}"
        #     self.logger.error(msg)
        #     raise ValueError(msg)

        # # Define the parameter grid
        # gp = self._run_gpr(actual_coords, haversine_errors, n_restarts_optimizer=50)

        # Calculate vector difference errors
        haversine_errors = haversine_distances_agg(
            actual_coords, predicted_coords, np.array
        )

        sigma = 0.01
        for i in range(2):
            actual_coords[:, i] = np.random.normal(actual_coords[:, i], sigma)

        # Spatial autocorrelation analysis
        # Adjust the threshold as needed. 1000 = 1 km
        weights = DistanceBand.from_array(actual_coords, threshold=1000.0)

        mi = Moran(haversine_errors, weights)

        # Run GPR with Matern kernel
        gp = self._run_gpr(actual_coords, haversine_errors, n_restarts_optimizer=50)

        # Create a grid over the area of interest
        grid_x, grid_y = np.meshgrid(
            np.linspace(
                actual_coords[:, 0].min() - buffer,
                actual_coords[:, 0].max() + buffer,
                1000,
            ),
            np.linspace(
                actual_coords[:, 1].min() - buffer,
                actual_coords[:, 1].max() + buffer,
                1000,
            ),
        )

        # Predict error and uncertainty for each point in the grid
        error_predictions, error_std = gp.predict(
            np.vstack((grid_x.ravel(), grid_y.ravel())).T, return_std=True
        )
        error_predictions = error_predictions.reshape(grid_x.shape)
        error_std = error_std.reshape(grid_x.shape)

        fig, axs = plt.subplots(1, 2, figsize=(16, 12))

        ax = axs[0]
        ax2 = axs[1]

        def roundup(x):
            x -= x % -100
            return x

        # Round to nearest 100.
        vmax = min(roundup(np.max(haversine_errors)), max_colorscale)
        vmax_std = min(roundup(np.max(error_std)), 100)

        # Define colormap and normalization
        # cmap = plt.get_cmap("plasma_r")

        tp_cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["#55CDFC", "#FFFFFF", "#F7A8B8"]
        )

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["#FF1B8D", "#FFDA00", "#1BB3FF"]
        )

        uncertainty_cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["#FF1B8D", "#FFDA00", "#1BB3FF"]
        )

        norm = mcolors.Normalize(vmin=min_colorscale, vmax=vmax)
        norm_std = mcolors.Normalize(vmin=0, vmax=vmax_std)

        contour = ax.contourf(
            grid_x,
            grid_y,
            error_predictions,
            cmap=cmap,
            norm=norm,
            levels=np.linspace(
                min_colorscale,
                vmax,
                num=n_contour_levels,
                endpoint=True,
            ),
            extend="max",
        )

        # uncertainty_cmap = plt.get_cmap("plasma_r")
        uncert_contour = ax2.contourf(
            grid_x,
            grid_y,
            error_std,
            cmap=uncertainty_cmap,
            norm=norm_std,
            levels=np.linspace(
                0,
                vmax_std,
                num=n_contour_levels,
                endpoint=True,
            ),
            extend="max",
        )

        cbar = self._make_colorbar(
            min_colorscale,
            vmax,
            n_contour_levels,
            ax,
            contour,
        )

        uncert_cbar = self._make_colorbar(
            0,
            vmax_std,
            n_contour_levels,
            ax2,
            uncert_contour,
        )

        # Error Vector Visualization (Quiver plot)
        # Calculate error vectors
        error_vectors = predicted_coords - actual_coords

        # # Normalize error vectors for visualization
        # norms = np.sqrt((error_vectors**2).sum(axis=1))
        # error_vectors_normalized = error_vectors / norms[:, np.newaxis]

        ax.quiver(
            actual_coords[:, 0],
            actual_coords[:, 1],
            error_vectors[:, 0],
            error_vectors[:, 1],
            color="red",
            scale=21,
            headwidth=3,
            headlength=5,
            headaxislength=4.5,
            alpha=0.7,
        )

        # Load and plot dynamic boundaries
        self._plot_map(actual_coords, url, self.output_dir, buffer, ax)
        self._plot_map(actual_coords, url, self.output_dir, buffer, ax2)

        # Create a GeoDataFrame for actual coordinates
        actual_centroids_transformed, actual_coords_transformed = self._get_points(
            actual_coords, predicted_coords, centroids
        )

        if dataset.lower() == "val":
            dataset = "validation"

        # Plot centroids layer.
        if centroids is not None:
            scatter2 = self._plot_scatter_map(
                dataset,
                ax,
                actual_centroids_transformed,
                marker_scale_factor,
                mult_factor=1.4,
                label="Centroids",
                color="k",
            )

        # Plot KDE layer
        scatter = self._plot_scatter_map(
            dataset, ax, actual_coords_transformed, marker_scale_factor
        )

        # Customization.
        self._set_geographic_plot_attr(ax)
        self._set_geographic_plot_attr(ax2)

        cbar = self._set_cbar_fontsize(cbar)
        uncert_cbar = self._set_cbar_fontsize(uncert_cbar)

        cbar.ax.set_title("Prediction Error (km)\n", fontsize=self.fontsize)
        uncert_cbar.ax.set_title("Interpolation Uncertainty\n", fontsize=self.fontsize)

        plt.subplots_adjust(wspace=0.5, hspace=0.05)

        ncol = 2 if centroids is not None else 1

        # Add legend
        ax.legend(
            bbox_to_anchor=(0.5, 1.7),
            loc="upper center",
            fontsize=self.fontsize,
            ncol=ncol,
        )

        # Ensure correct scale and aspect ratio
        ax.set_aspect("equal", "box")
        ax2.set_aspect("equal", "box")

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_geographic_error_{dataset}.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def _set_geographic_plot_attr(self, ax):
        """
        Sets the common attributes for geographic plots, including labels for longitude and latitude.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib Axes object for the plot.

        Notes:
            - This is a helper method used internally to standardize the appearance of geographic plots.
        """
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    def _run_gpr(self, actual_coords, haversine_errors, n_restarts_optimizer=50):
        """
        Runs Gaussian Process Regression (GPR) on actual coordinates against Haversine errors to model the spatial distribution of prediction errors.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            haversine_errors (np.array): Array of Haversine errors between actual and predicted coordinates.
            n_restarts_optimizer (int, optional): Number of restarts for the optimizer in GPR. Defaults to 25.

        Returns:
            GaussianProcessRegressor: The fitted Gaussian Process Regressor model.

        Notes:
            - The method defines and fits a GPR model with a specific kernel to capture the spatial variability of errors.
        """
        # Define the kernel with parameter ranges
        # Fit Gaussian Process Regressor with a larger initial length scale and
        # no upper bound
        kernel = 1 * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e6)
        ) + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-10, 1e6))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=custom_gpr_optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
        )
        gp.fit(actual_coords, haversine_errors)
        return gp

    def _set_cbar_fontsize(self, cbar):
        """
        Sets the font size for the colorbar labels.

        Args:
            cbar (matplotlib.colorbar.Colorbar): The colorbar object whose font size is to be set.

        Returns:
            matplotlib.colorbar.Colorbar: The colorbar object with updated font size.

        Notes:
            - This is a utility method for adjusting the appearance of colorbars in plots.
        """
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(self.fontsize)
        return cbar

    def _plot_scatter_map(
        self,
        dataset,
        ax,
        coords,
        exp_factor,
        mult_factor=1.0,
        label="Samples",
        alpha=0.5,
        color="darkorchid",
    ):
        """
        Plots a scatter map of coordinates, with the size of each point representing a certain attribute (e.g., sample weight).

        Args:
            dataset (str): Name of the dataset being used.
            ax (matplotlib.axes.Axes): The matplotlib Axes object for the plot.
            coords (np.array): Array of coordinates to be plotted.
            exp_factor (int): Exponent factor for scaling the size of the markers.
            mult_factor (float, optional): Multiplicative factor for marker size. Defaults to 1.0.
            label (str, optional): Label for the plotted points. Defaults to "Samples".
            alpha (float, optional): Alpha transparency for the markers. Defaults to 0.5.
            color (str, optional): Color of the markers. Defaults to "darkorchid".

        Returns:
            matplotlib.collections.PathCollection: The scatter plot object.

        Notes:
            - This method is used for creating scatter plots on geographical maps with customizable marker sizes and colors.
        """
        return ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=plt.rcParams["lines.markersize"] ** exp_factor * mult_factor,
            c=color,
            alpha=alpha,
            label=f"{dataset.capitalize()} {label}",
        )

    def _get_points(self, actual_coords, predicted_coords, centroids):
        """
        Transforms actual and predicted coordinates to a suitable projection for plotting and optionally
        includes centroids if provided.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            predicted_coords (np.array): Array of predicted geographical coordinates.
            centroids (np.array, optional): Array of centroids to be included. Defaults to None.

        Returns:
            tuple: A tuple containing transformed arrays of actual coordinates, predicted coordinates, and centroids.

        Notes:
            - The method converts the coordinates to a GeoDataFrame and then to the WGS 84 coordinate system for plotting.
        """
        gdf_actual = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in actual_coords], crs="EPSG:4326"
        )

        # Create a GeoDataFrame for predicted coordinates
        gdf_predicted = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in predicted_coords], crs="EPSG:4326"
        )

        actual_centroids_transformed = None
        if centroids is not None:
            gdf_centroids = gpd.GeoDataFrame(
                geometry=[Point(xy) for xy in centroids], crs="EPSG:4326"
            )
            gdf_centroids = gdf_centroids.to_crs(epsg=4326)
            actual_centroids_transformed = np.array(
                [(point.x, point.y) for point in gdf_centroids.geometry]
            )

        # Transform to WGS 84
        gdf_actual = gdf_actual.to_crs(epsg=4326)
        gdf_predicted = gdf_predicted.to_crs(epsg=4326)

        actual_coords_transformed = np.array(
            [(point.x, point.y) for point in gdf_actual.geometry]
        )

        return actual_centroids_transformed, actual_coords_transformed

    def _make_colorbar(
        self, min_colorscale, max_colorscale, n_contour_levels, ax, contour
    ):
        """
        Creates and configures a colorbar for contour plots.

        Args:
            min_colorscale (int): Minimum value for the color scale.
            max_colorscale (int): Maximum value for the color scale.
            n_contour_levels (int): Number of contour levels in the plot.
            ax (matplotlib.axes.Axes): The matplotlib Axes object for the plot.
            contour (matplotlib.contour.QuadContourSet): The contour plot object.

        Returns:
            matplotlib.colorbar.Colorbar: The colorbar object.

        Notes:
            - This method sets up a colorbar with specified min and max values, and a defined number of levels.
        """
        cbar = plt.colorbar(contour, ax=ax, extend="max", fraction=0.046, pad=0.1)

        cbar.set_ticks(
            np.linspace(
                min_colorscale, max_colorscale, num=n_contour_levels // 2, endpoint=True
            )
        )

        cbar.set_ticklabels(
            [
                str(int(x))
                for x in np.linspace(
                    min_colorscale,
                    max_colorscale,
                    num=n_contour_levels // 2,
                    endpoint=True,
                )
            ]
        )

        return cbar

    def _plot_map(self, actual_coords, url, output_dir, buffer, ax):
        """
        Plots a base map using shapefile data from a specified URL or file path.

        Args:
            actual_coords (np.array): Array of geographical coordinates to plot.
            url (str): URL to download the shapefile data.
            output_dir (str): Directory where the shapefile data will be saved.
            buffer (float): Buffer distance to adjust the plotted area around the coordinates.
            ax (matplotlib.axes.Axes): The matplotlib Axes object for the plot.

        Returns:
            matplotlib.axes.Axes: The updated Axes object with the base map plotted.

        Notes:
            - This method downloads the shapefile data if not already present and plots it as a base map.
            - The buffer parameter allows for adjusting the view area around the plotted coordinates.
        """
        # Ensure coordinates are valid
        if np.any(np.isnan(actual_coords)) or np.any(np.isinf(actual_coords)):
            self.logger.error("Invalid coordinates detected.")
            raise ValueError("Invalid coordinates in actual_coords.")

        mapdata = self._extract_basemap_path_url(url, output_dir)

        # Set the limits with a buffer
        x_min, x_max = actual_coords[:, 0].min(), actual_coords[:, 0].max()
        y_min, y_max = actual_coords[:, 1].min(), actual_coords[:, 1].max()

        x_min -= buffer
        y_min -= buffer
        x_max += buffer
        y_max += buffer

        # Plotting
        mapdata = mapdata.clip([x_min, y_min, x_max, y_max])

        mapdata.plot(
            ax=ax,
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ Country Lines",
        )

        return ax

    def _extract_basemap_path_url(self, url, output_dir=None):
        """Extract base map from provided URL or file path.

        Args:
            url (str): Either a valid URL or file path.
            output_dir (str): Output directory to save base map to, if downloading from URL. Defaults to None.

        Returns:
            geopandas.GeoDataFrame: Base map as geopandas dataframe.

        """
        if output_dir is None:
            output_dir = self.output_dir

        outshp = Path(output_dir)

        # Check if is url or is file path.
        is_url = False if urlparse(url).scheme == "" else True
        fn = url.split("/")[-1] if is_url else Path(url)

        if not is_url:
            fn = fn.name

        dest = outshp.joinpath("shapefile", fn)

        if is_url and not dest.exists():
            try:
                wget.download(url, outshp, bar=None)
            except Exception as e:
                msg = f"Error downloading shapefile from {url}: {e}."
                self.logger.error(msg)
                raise e

        if not is_url and not Path(url).is_file():
            msg = f"Could not find shapefile in provided path: {mapfile}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        # Extract shapefile directory, if necessary.
        mapfile = self._extract_shapefile(url)

        try:
            mapdata = gpd.read_file(mapfile)
        except Exception as e:
            self.logger.error(f"Could not read map file {mapfile} from {url}.")
            raise e

        mapdata.crs = "epsg:4326"

        if self.basemap_fips is not None:
            mapdata = mapdata[mapdata["STATEFP"] == self.basemap_fips]
        return mapdata

    def plot_cumulative_error_distribution(self, data, fn, percentiles, median, mean):
        """
        Generate an ECDF plot for the given data.

        Args:
            data (array-like): The dataset for which the ECDF is to be plotted.
            fn (str): Output filename.
            percentiles (np.ndarray): 25th, 50th, and 75th percentiles of errors. Will be of shape (3,).
            median (float): Median of prediction errors.
            mean (float): Mean of prediction errors.

        Returns:
            matplotlib.figure.Figure: The ECDF plot.
        """
        # Sort the data
        x = np.sort(data)

        def roundup(x):
            x -= x % -100
            return x

        plt.figure(figsize=(12, 12))

        # Colormap and normalization
        vmax = min(roundup(np.max(x)), 150)

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["#FF1B8D", "#FFDA00", "#1BB3FF"]
        )

        # cmap = plt.get_cmap("plasma_r")
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        # Create the plot
        ax = sns.histplot(
            x,
            fill=False,
            stat="percent",
            cumulative=True,
            kde=True,
            line_kws={
                "lw": 8,
            },
            bins=25,
            color="none",
            edgecolor="k",
        )

        # Add Gradient Fill
        self._fill_kde_with_gradient(x, cmap, norm, ax)

        plt.xlabel("Haversine Error (km)")
        plt.ylabel("Cumulative Percent of Distribution")
        plt.title("Empirical Cumulative Distribution Function (ECDF)")

        plist = ["25th", "50th", "75th"]
        clrs = ["dodgerblue", "gold", "darkorchid"]

        for i, perc in enumerate(percentiles):
            if i != 1:
                plt.axvline(
                    perc,
                    label=f"{plist[i]} Percentile",
                    color=clrs[i],
                    linestyle="dashed",
                    lw=4,
                )

        plt.axvline(
            np.round(mean, decimals=1),
            label="Mean",
            color="lightseagreen",
            linestyle="solid",
            lw=4,
        )

        plt.axvline(
            median, label="Median", color="darkorange", linestyle="dashdot", lw=4
        )
        plt.legend(
            loc="upper left",
            bbox_to_anchor=(1.04, 1),
            fancybox=True,
            shadow=True,
            borderpad=1,
        )

        if self.show_plots:
            plt.show()

        outfile = self.outbasepath.with_name(fn)
        outfile = outfile.with_suffix("." + self.filetype)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def _fill_kde_with_gradient(self, xdata, cmap, norm, ax=None, ydata=None):
        """Fill a KDE plot with a gradient following the X-axis values.

        Args:
            xdata (np.ndarray): X-axis values to plot.
            cmap (matplotlib.colors.cmap): Matplotlib colormap to use.
            norm (matplotlob.colors.Normalize): Normalizer for color gradient.
            ax (matplotlib.pyplot.Axes or None): Matplotlib axis to use. If ydata is None, then ax must be provided. Defaults to None.
            ydata (np.ndarray): Y-axis values to plot. If None, then gets the y-axis values from the provided `ax` object. Defaults to None.)
        """
        if ydata is None and ax is None:
            self.logger.error("ax must be defined if ydata is None.")
            raise TypeError("ax must be defined if ydata is None.")

        if ydata is None:
            lines2d = [
                obj
                for obj in ax.findobj()
                if str(type(obj)) == "<class 'matplotlib.lines.Line2D'>"
            ]
            xdata, ydata = lines2d[0].get_data()

        for i in range(len(xdata) - 1):
            plt.fill_between(
                xdata[i : i + 2],
                ydata[i : i + 2],
                color=cmap(norm(xdata[i])),
            )

    def plot_zscores(self, z, errors, fn):
        """Plot Z-score histogram for prediction errors.

        Args:
            z (np.ndarray): Array of Z-scores.
            errors (np.ndarray): Array of prediction errors.
            fn (str): Filename for the output plot.
        """
        plt.figure(figsize=(12, 12))

        cmap = plt.colormaps.get_cmap("Purples")
        norm = mcolors.Normalize(vmin=np.min(z), vmax=np.max(z))

        ax = sns.histplot(
            x=z,
            stat="proportion",
            bins=25,
            fill=False,
            kde=True,
            cumulative=False,
            color="none",
            edgecolor="none",
            line_kws={"lw": 4, "color": "k", "alpha": 0.75},
        )

        line = ax.lines[0]
        xdata, ydata = line.get_data()

        for i in range(len(xdata) - 1):
            plt.fill_between(
                xdata[i : i + 2],
                ydata[i : i + 2],
                color=cmap(norm(xdata[min(i + 8, len(xdata) - 1)])),
            )

        plt.xlabel("Z-Scores")
        plt.ylabel("Proportion")
        plt.title("Z-Score Distribution of Prediction Errors")

        # Annotations and Highlights
        mean_z = np.mean(z)
        plt.axvline(
            np.round(mean_z, decimals=1),
            label="Mean",
            color="darkorange",
            lw=4,
            linestyle="--",
        )

        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))

        if self.show_plots:
            plt.show()

        outfile = self.outbasepath.with_name(fn)
        outfile = outfile.with_suffix("." + self.filetype)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_error_distribution(self, errors, outfile):
        """
        Plot the distribution of errors using a histogram, box plot, and Q-Q plot.

        Args:
            errors (np.array): An array of prediction errors.
            outfile (str): Output file path.

        Returns:
            None: Plots the error distribution.
        """
        plt.figure(figsize=(18, 6))

        def roundup(x):
            x -= x % -100
            return x

        vmax = min(roundup(np.max(errors)), 150)

        # Colormap and normalization
        # cmap = plt.get_cmap("plasma_r")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "", ["#FF1B8D", "#FFDA00", "#1BB3FF"]
        )
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        # Histogram (Density Plot with Gradient)
        plt.subplot(1, 3, 1)

        # Compute KDE
        kde = stats.gaussian_kde(errors)
        x_values = np.linspace(np.min(errors), np.max(errors), 150)
        kde_values = kde(x_values)

        # Normalize the KDE values
        # kde_values /= np.sum(kde_values)

        # Create Line Plot for the KDE
        plt.plot(x_values, kde_values, color="black")

        # Add Gradient Fill
        self._fill_kde_with_gradient(x_values, cmap, norm, ydata=kde_values)

        plt.title("Prediction Error x Sampling Density")
        plt.xlabel("Haversine Error (km)")
        plt.ylabel("Sampling Density")

        # Box Plot
        plt.subplot(1, 3, 2)
        bplot = plt.boxplot(
            errors,
            vert=False,
            notch=True,
            bootstrap=1000,
            patch_artist=True,
            showfliers=True,
        )
        plt.title("Prediction Error Box Plot")
        plt.xlabel("Haversine Error (km)")
        plt.ylabel("")

        for patch, color in zip(bplot["boxes"], ["darkorchid"]):
            patch.set_facecolor(color)

        # Q-Q Plot
        plt.subplot(1, 3, 3)
        stats.probplot(errors, dist="norm", plot=plt, rvalue=True, fit=True)
        plt.title("Quantile x Quantile Error")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Actual Quantiles")

        if self.show_plots:
            plt.show()
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def polynomial_regression_plot(
        self, actual_coords, predicted_coords, dataset, degree=3, dtype=torch.float32
    ):
        """
        Creates a polynomial regression plot with the specified degree.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            predicted_coords (np.array): Array of predicted geographical coordinates by the model.
            dataset (str): Specifies the dataset being used, should be either 'test' or 'validation'.
            degree (int): Polynomial degree to fit. Defaults to 3.
            dtype (torch.dtype): PyTorch data type to use. Defaults to torch.float32.

        Raises:
            ValueError: If the dataset parameter is not 'test' or does not start with 'val'.

        Notes:
            - This function calculates the Haversine error for each pair of actual and predicted coordinates.
            - It then computes the KDE values for these errors and plots a regression to analyze the relationship.
        """
        if (
            dataset != "test"
            and not dataset.startswith("val")
            and not dataset == "train"
        ):
            msg = (
                f"'dataset' parameter must be either 'test' or 'validation': {dataset}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        sampler = GeographicDensitySampler(
            pd.DataFrame(actual_coords, columns=["x", "y"]),
            use_kde=True,
            use_kmeans=False,
            max_clusters=10,
            max_neighbors=50,
            verbose=0,
            dtype=dtype,
        )

        x = sampler.density

        # Calculate Haversine error for each pair of points
        y = haversine_distances_agg(actual_coords, predicted_coords, np.array)

        plt.figure(figsize=(12, 12))

        # Create polynomial features
        poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Fit to data
        poly_model.fit(x[:, np.newaxis], y)

        # Generate predictions for plotting
        xfit = np.linspace(np.min(x), np.max(x), 1000)
        yfit = poly_model.predict(xfit[:, np.newaxis])

        r, p = stats.pearsonr(x, y)
        if p < 0.0001:
            p = "< 0.0001"
        elif p < 0.001:
            p = "< 0.001"
        else:
            p = f"= {p:.2f}"

        sns.regplot(
            x=xfit,
            y=yfit,
            fit_reg=True,
            n_boot=1000,
            line_kws={"lw": 5, "color": "darkorchid"},
            label=f"Pearson's R = {r:.2f}\nP-value {p}",
        )

        # Plotting the results
        plt.scatter(
            x,
            y,
            alpha=0.7,
            color="lightseagreen",
            label="Samples",
            s=plt.rcParams["lines.markersize"] ** 2 * 5,
            lw=2,
            edgecolors="k",
        )

        plt.ylabel("Prediction Error (km)")
        plt.xlabel("Sampling Density")
        plt.title("Polynomial Regression of Kernel Density vs Prediction Error (km)")

        plt.legend(
            loc="upper left", bbox_to_anchor=(1.04, 1.0), fancybox=True, shadow=True
        )

        # Show and save plot
        self._show_and_save_plot(plt, dataset)

    def plot_kde_error_regression(self, actual_coords, predicted_coords, dataset):
        """
        Plots a regression analysis between the kernel density estimation (KDE) density and the prediction error of geographical coordinates. This method helps in understanding the relationship between sampling density (as estimated by KDE) and the accuracy of predictions.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            predicted_coords (np.array): Array of predicted geographical coordinates by the model.
            dataset (str): Specifies the dataset being used, should be either 'test' or 'validation'.

        Raises:
            ValueError: If the dataset parameter is not 'test' or does not start with 'val'.

        Notes:
            - This function calculates the Haversine error for each pair of actual and predicted coordinates.
            - It then computes the KDE values for these errors and plots a regression to analyze the relationship.
        """
        if dataset != "test" and not dataset.startswith("val"):
            msg = (
                f"'dataset' parameter must be either 'test' or 'validation': {dataset}."
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Calculate Haversine error for each pair of points
        haversine_errors = haversine_distances_agg(
            actual_coords, predicted_coords, np.array
        )

        # Compute KDE with adjusted bandwidth if necessary
        bandwidth = np.std(haversine_errors) * len(haversine_errors) ** (-1 / 5.0)
        kde = stats.gaussian_kde(haversine_errors, bw_method=bandwidth)
        kde_values = kde.evaluate(haversine_errors)

        # Data transformation and DataFrame creation
        df = pd.DataFrame(
            {
                "Sampling Density": kde_values,
                "Prediction Error": haversine_errors,
            }
        )

        # Calculate R-squared for regression
        r2 = stats.pearsonr(df["Sampling Density"], df["Prediction Error"])[0] ** 2

        # Regression plot
        plt.figure(figsize=(10, 6))
        ax = sns.regplot(
            x="Sampling Density",
            y="Prediction Error",
            data=df,
            scatter=True,
            scatter_kws={"color": "darkorchid", "alpha": 0.5},
            line_kws={"color": "darkorchid", "lw": 3},
            robust=True,
            n_boot=1000,
            label=f"R²: {r2:.2f}",
        )
        ax.legend(fontsize=self.fontsize)
        ax.set_xlabel("Log Sampling Density")
        ax.set_ylabel("Prediction Error")
        ax.set_title("Regression of Kernel Density vs Prediction Error")

        # Show and save plot
        self._show_and_save_plot(plt, dataset)

    def _show_and_save_plot(self, plt, dataset):
        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_kde_error_regression_{dataset}.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_mca_curve(self, explained_inertia, optimal_n):
        """
        Plots the cumulative explained inertia as a function of the number of components in Multiple Correspondence Analysis (MCA).

        This plot is useful for determining the optimal number of components to retain in MCA.

        Args:
            explained_inertia (array-like): An array of cumulative explained inertia for each number of components.
            optimal_n (int): The optimal number of components determined for MCA.

        Notes:
            - The plot displays the explained inertia against the number of components.
            - A vertical line indicates the selected optimal number of components.
        """
        # Function implementation...
        plt.figure(figsize=(12, 12))
        plt.plot(
            range(1, len(explained_inertia) + 1),
            explained_inertia,
            linestyle="-",
            lw=3,
            c="darkorchid",
        )
        plt.xlabel("Number of Components")
        plt.ylabel("Explained Inertia")
        plt.title("MCA Explained Inertia")
        plt.axvline(x=optimal_n, color="darkorange", linestyle="--")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_mca_curve.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_nmf_error(self, errors, opt_n_components):
        """
        Plots the reconstruction error as a function of the number of components in Non-negative Matrix Factorization (NMF).

        This plot can be used to select the optimal number of components for NMF by identifying the point where additional components do not significantly decrease the error.

        Args:
            errors (list): A list of NMF reconstruction errors for each number of components.
            opt_n_components (int): The optimal number of components selected for NMF.

        Notes:
            - The plot visualizes how the reconstruction error changes with the number of NMF components.
            - A vertical line indicates the selected optimal number of NMF components.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        x = range(1, len(errors) + 1)
        y = errors.copy()
        ax.plot(x, y, "-", color="darkorchid", lw=3)
        ax.set_xlabel("NMF Components")
        ax.set_ylabel("NMF Reconstruction Error")
        ax.axvline(x=opt_n_components, label="Selected Number of NMF Components")
        ax.legend(loc="best")
        ax.set_title("NMF Components vs. Reconstruction Error")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_nmf_curve.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_pca_curve(self, x, vr, knee):
        """
        Plots the cumulative explained variance as a function of the number of principal components in Principal Component Analysis (PCA).

        This plot is helpful for determining the number of components to retain in PCA.

        Args:
            x (array-like): An array representing the number of principal components.
            vr (array-like): An array of cumulative explained variance ratios for each number of components.
            knee (int): The 'knee' point, or the optimal number of components to retain in PCA.

        Notes:
            - The plot shows the cumulative explained variance against the number of principal components.
            - A vertical line at the 'knee' point helps in visually identifying the optimal number of components.
        """
        plt.figure(figsize=(12, 12))
        plt.plot(x, vr, "-", color="darkorchid")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.axvline(
            x=knee,
            label="Selected Number of Components",
            linestyle="--",
            color="orange",
        )
        plt.legend(loc="best")
        plt.title(f"Principal Components vs. Explained Variance")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_pca_curve.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_dbscan_clusters(self, Xy, dataset, labels, url, buffer=1.0):
        """
        Plots the clusters formed by DBSCAN algorithm on geographical data. Each cluster is visualized with a different color, and outliers are marked distinctly.

        Args:
            Xy (np.array): Array containing the data with pre-transformed 'x' and 'y' coordinates.
            dataset (str): Name of the dataset being used.
            labels (np.array): Cluster labels assigned by DBSCAN to each data point.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 1.0.

        Notes:
            - The function converts the data to a GeoDataFrame for geographical plotting.
            - Different clusters are visualized in different colors, with outliers typically in red.
        """
        # Applying DBSCAN
        df = pd.DataFrame(Xy, columns=["x", "y"] + list(range(Xy.shape[1] - 2)))

        # Convert to GeoDataFrame for plotting
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
        gdf["cluster"] = labels

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        ax = self._plot_map(Xy, url, self.output_dir, buffer, ax)

        # Plot each cluster with different color
        unique_labels = set(labels)
        for label in unique_labels:
            cluster_gdf = gdf[gdf["cluster"] == label]
            color = (
                "red"
                if label == -1
                else np.random.rand(
                    3,
                )
            )  # Outliers in red
            cluster_gdf.plot(ax=ax, color=color)

        if self.show_plots:
            plt.show()

        fn = (f"{self.prefix}_outlier_clustering_{dataset}.{self.filetype}",)
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_geographical_heatmap(
        self, data, weights, url, buffer=0.1, title="Sampling Weight Heatmap"
    ):
        """
        Plots a geographical heatmap representing the sampling weights of data points. The heatmap provides a visual representation of the density or weight of data points in geographical space.

        Args:
            data (pandas DataFrame): DataFrame containing 'longitude' and 'latitude' columns for geographical data points.
            weights (np.ndarray): Array of weights corresponding to each data point in the DataFrame.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.
            title (str, optional): Title for the heatmap plot. Defaults to "Sampling Weight Heatmap".

        Notes:
            - The function creates a GeoDataFrame from the provided data and weights for plotting.
            - The heatmap uses color intensity to represent the density or weight of data points in different geographical locations.
        """

        gdf = gpd.GeoDataFrame(
            data, geometry=gpd.points_from_xy(data[:, 0], data[:, 1])
        )
        gdf["weights"] = weights

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        self._plot_map(data, url=url, output_dir=self.output_dir, buffer=buffer, ax=ax)
        gdf.plot(
            column="weights",
            ax=ax,
            legend=True,
            cmap="viridis",
            markersize=20,
        )

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_train_sample_density_heatmap.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_weight_distribution(self, weights, title="Weight Distribution"):
        """
        Plots the distribution of sampling weights as a histogram. This visualization helps in understanding the distribution and range of weights assigned to the samples.

        Args:
            weights (np.ndarray): Array of weights.
            title (str, optional): Title for the histogram plot. Defaults to "Weight Distribution".

        Notes:
            - The histogram displays the frequency of different weight values among the samples.
        """
        plt.figure(figsize=(8, 5))
        plt.hist(weights, bins=30, alpha=0.7)
        plt.title(title)
        plt.xlabel("Weight")
        plt.ylabel("Frequency")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_sample_weight_distribution.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_weighted_scatter(
        self,
        data,
        weights,
        url,
        buffer=0.1,
        marker_scale_factor=3,
        title="Geographic Scatter Plot",
    ):
        """
        Plots a geographic scatter plot where each data point is sized according to its weight. This plot visually represents the relative importance or weight of each data point in a geographical context.

        Args:
            data (pandas DataFrame): DataFrame containing 'longitude' and 'latitude' columns.
            weights (np.ndarray): Array of weights corresponding to each data point.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.
            marker_scale_factor (int, optional): Factor to scale the size of markers in the scatter plot. Defaults to 3.
            title (str, optional): Title for the scatter plot. Defaults to "Geographic Scatter Plot".

        Notes:
            - The size of each marker in the scatter plot is proportional to the corresponding weight of the data point.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        self._plot_map(data, url, self.output_dir, buffer, ax)

        ax = sns.scatterplot(
            x=data[:, 0],
            y=data[:, 1],
            size=weights,
            sizes=(
                plt.rcParams["lines.markersize"] ** marker_scale_factor * 4,
                plt.rcParams["lines.markersize"] ** marker_scale_factor * 16,
            ),
            alpha=0.5,
            c="darkorchid",
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_sample_weight_scatterplot.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_outliers(self, mask, y_true, url, buffer=0.1):
        """
        Plots a scatter plot to visualize the identified outliers in the dataset. Outliers are marked distinctly to
        differentiate them from the regular data points.

        Args:
            mask (np.array): A boolean array where 'True' indicates an outlier.
            y_true (np.array): Array of actual coordinates.
            url (str): URL for the shapefile to plot geographical data.
            buffer (float, optional): Buffer distance for geographical plotting. Defaults to 0.1.

        Notes:
            - The function visualizes outliers on a geographical map, aiding in the identification of anomalous data points.
        """
        df = pd.DataFrame(y_true, columns=["x", "y"])
        df["Outliers"] = ~mask
        df["Outliers"] = df["Outliers"].astype(str)
        df = df[~df["x"].isna()]
        df["Sizes"] = "Non-Outlier"
        df.loc[df["Outliers"] == "True", "Sizes"] = "Outlier"

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        self._plot_map(df[["x", "y"]].to_numpy(), url, self.output_dir, buffer, ax)

        ax = sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="Outliers",
            size="Sizes",
            sizes=(100, 1000),
            size_order=["Outlier", "Non-Outlier"],
            palette="Set2",
            alpha=0.7,
            ax=ax,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Outliers Removed from Dataset")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.04, 1),
            markerscale=1,
        )

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_outlier_scatterplot.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_outliers_with_traces(
        self,
        genetic_data,
        geographic_data,
        genetic_outliers,
        geographic_outliers,
        correct_centroids_gen,
        correct_centroids_geo,
        url,
        buffer=0.5,
    ):
        """
        Plots geographic data with traces to centroids for outliers only if they are in the wrong cluster.

        Args:
            geographic_data (numpy.ndarray): The geographic data of the samples.
            genetic_outliers (set): Indices of genetic outliers.
            geographic_outliers (set): Indices of geographic outliers.
            cluster_centroids_gen (dict): Genetic cluster centroids.
            cluster_centroids_geo (dict): Geographic cluster centroids.
            current_cluster_assignments (numpy.ndarray): Current cluster assignments for each sample.
            url (str): URL for map data.
            buffer (float): Buffer size for the map plot.
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        for ax, data_type in zip(axs, ["geographic", "genetic"]):
            data = genetic_data if data_type == "genetic" else geographic_data
            if data_type == "geographic":
                ax = self._plot_map(data, url, self.output_dir, buffer, ax)

            # Plot all samples
            ax.scatter(data[:, 0], data[:, 1], alpha=0.5)

            # Function to draw a line from sample to the correct cluster
            # centroid
            def draw_line_to_correct_centroid(idx, centroid, color):
                ax.plot(
                    [data[idx][0], centroid[0]],
                    [data[idx][1], centroid[1]],
                    color=color,
                    alpha=0.5,
                )

            # Draw lines for misclustered samples
            for idx in geographic_outliers:
                if idx in correct_centroids_geo:
                    draw_line_to_correct_centroid(
                        idx,
                        correct_centroids_geo[idx],
                        "red",
                    )

            for idx in genetic_outliers:
                if idx in correct_centroids_gen:
                    draw_line_to_correct_centroid(
                        idx,
                        correct_centroids_gen[idx],
                        "blue",
                    )

            if data_type == "geographic":
                xlab = "Longitude"
                ylab = "Latutude"
            else:
                xlab = "Principal Component 1"
                ylab = "Principal Component 2"
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_outliers.png"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, dtype):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            ax=ax,
        )

        if self.show_plots:
            plt.show()
        fn = f"{self.prefix}_outlier_{dtype}_confusion_matrix.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white")
        plt.close()

    def plot_gamma_distribution(self, shape, scale, Dg, sig_level, filename, plot_main):
        """
        Plot the gamma distribution.

        Args:
            shape (float): Shape parameter of the gamma distribution.
            scale (float): Scale parameter of the gamma distribution.
            Dg (np.array): Dg statistic for each sample.
            sig_level (float): Significance level (e.g., 0.05).
            filename (str): Name of the file to save the plot.
            plot_main (str): Title of the plot.
        """
        x = np.linspace(0, np.max(Dg), 1000)
        y = gamma.pdf(x, a=shape, scale=scale)

        gamma_threshold = gamma.ppf(1 - sig_level, a=shape, scale=scale)

        plt.figure(figsize=(16, 12))
        plt.plot(x, y, color="blue")

        plt.ylim(0, max(1, np.max(y)))
        plt.axvline(
            x=gamma_threshold,
            color="darkorange",
            linestyle="--",
            lw=3,
            label=f"P = {sig_level}",
        )
        plt.xlabel(f"Gamma(α={shape:.2f}, β={scale:.2f})")
        plt.ylabel("Density")
        plt.title(plot_main)

        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1.0))

        if self.show_plots:
            plt.show()
        plt.savefig(filename, facecolor="white", bbox_inches="tight")
        plt.close()

    def _calculate_centroid(self, gdf):
        """
        Calculate the centroid of a set of geographic points.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame containing point geometries.

        Returns:
            gpd.GeoSeries: Centroids.
        """
        return gdf.geometry.unary_union.centroid

    def _extract_shapefile(self, zip_path):
        """
        Extract a zipped shapefile.

        Args:
            zip_path (str): Path to the zipped shapefile.

        Returns:
            str: Path to the extracted shapefile directory, which will be a temporary directory, or if the directory was already unzipped then just returns the original file path.
        """
        if zip_path.endswith(".zip"):
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()

            # Extract the zip file
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            return temp_dir
        return zip_path

    def plot_boot_ci_data(
        self, df, stats_df, sample_id, shapefile, known_locality=None
    ):
        """
        Plot bootstrapped locality data with mean, confidence intervals, and DRMS.

        Args:
            df (pd.DataFrame): DataFrame containing 'x' and 'y' coordinates (decimal degrees), plus summary stats for the bootstrap replicates.
            stats_df (dict): Summary statistics containing the means, std_dev, and 95% CIs for 'x' and 'y'.
            sample_id (str): Sample ID to use in output filename.
            shapefile (str): Shapefile path or URL. Can be zipped or unzipped.
            known_locality (str): Path to file containing known localities for prediction data. If undefined, then no recorded locality marker is plotted. Defaults to None.
        """
        gray_counties = self.basemap_highlights

        df_known = pd.read_csv(known_locality, sep="\t")
        df_known = df_known[df_known["sampleID"] == sample_id]
        sample_gdf = self._get_known_pred_locations(df)
        gdf_known = self._get_known_pred_locations(df_known)

        gdf = self._extract_basemap_path_url(shapefile)

        coords = np.array(list(sample_gdf.geometry.apply(lambda p: (p.x, p.y))))

        # Calculate Mean Coordinates
        centroids = self._calculate_centroid(sample_gdf)

        if known_locality is not None:
            centroid_known = self._calculate_centroid(gdf_known)

        mean_lon, mean_lat = centroids.x, centroids.y

        # Calculate bounds with buffer
        xmin, ymin, xmax, ymax = sample_gdf.total_bounds

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot base map.
        gdf.plot(ax=ax, color="none", edgecolor="black")
        gdf = self._hightlight_counties(gray_counties, gdf, ax)

        # Apply Gaussian Kernel Density Estimation
        kde = stats.gaussian_kde(coords.T)

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

        if known_locality is not None:
            ax.scatter(
                centroid_known.x,
                centroid_known.y,
                s=100,
                color="k",
                marker="^",  # triangle symbol
                label="Recorded Locality",
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{sample_id}: Bootstrapped Locality Predictions")

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

        if self.show_plots:
            plt.show()

        pth = Path(self.output_dir)
        od = pth.joinpath("plots", "sample_prediction_plots")
        od.mkdir(exist_ok=True, parents=True)
        of = f"{self.prefix}_bootstrap_confidence_plot_{sample_id}.{self.filetype}"
        fn = od.joinpath(of)

        fig.savefig(fn, bbox_inches="tight", facecolor="white")
        plt.close()

    def _plot_single_sample_boot_error(
        self,
        sample_gdf,
        sample_id,
        shapefile,
        sample_gdf_known=None,
    ):
        """
        Plot the bootstrapped points using a Gaussian KDE, including contour lines for density, and highlight the centroid for a given sample on a basemap with US state county lines.

        Args:
            sample_gdf (gpd.GeoDataFrame): GeoDataFrame containing coordinates for a single sample.
            sample_id (str): Sample ID to plot.
            shapefile (str): Directory or URL containing the county shapefile. Shapefile should be zipped.
            sample_gdf_known (GeoPandas.DataFrame): GeoPandas DataFrame with known coordinates for predicted values. Defaults to None.
        """
        gray_counties = self.basemap_highlights

        gdf = self._extract_basemap_path_url(shapefile)

        coords = np.array(list(sample_gdf.geometry.apply(lambda p: (p.x, p.y))))

        # Calculate Mean Coordinates
        centroids = self._calculate_centroid(sample_gdf)

        if sample_gdf_known is not None:
            centroid_known = self._calculate_centroid(sample_gdf_known)

        mean_lon, mean_lat = centroids.x, centroids.y

        # Calculate bounds with buffer
        xmin, ymin, xmax, ymax = sample_gdf.total_bounds

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot base map.
        gdf.plot(ax=ax, color="none", edgecolor="black")
        gdf = self._hightlight_counties(gray_counties, gdf, ax)

        # Apply Gaussian Kernel Density Estimation
        kde = stats.gaussian_kde(coords.T)

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
                marker="^",  # triangle symbol
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

        pth = Path(self.output_dir)
        od = pth.joinpath("plots", "sample_prediction_plots")
        od.mkdir(exist_ok=True, parents=True)
        of = f"{self.prefix}_bootstrap_prediction_{sample_id}.{self.filetype}"
        fn = od.joinpath(of)

        plt.savefig(fn, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _hightlight_counties(self, gray_counties, gdf, ax=None):
        if gray_counties is not None:
            # Filtering the counties to be colored gray
            gray_county_gdf = gdf[gdf["NAME"].isin(gray_counties)]
            gray_county_gdf.plot(ax=ax, color="darkgray", edgecolor="k", alpha=0.5)
            return gray_county_gdf
        return gdf

    def plot_samples_boot_error(
        self,
        directory,
        basemap,
        selected_samples=None,
        known_coords=None,
    ):
        """
        Function to process and plot data for selected samples.

        Args:
            directory (str): Directory containing CSV files.
            basemap (str): Path to the zipped basemap shapefile or a URL to download the base map.
            selected_samples (list of str, optional): List of sampleIDs to be plotted. Plots all if None.
            known_coords (str): Path to known coordinates for predicted samples. For validation. If None, then doesn't plot the real coordinates. Defaults to None.
        """
        gdf = load_directory_of_locs(directory)
        df_known = pd.read_csv(known_coords, sep="\t")

        for sample_id in gdf["sampleID"].unique():
            if selected_samples is None or sample_id in selected_samples:
                sample_gdf = gdf[gdf["sampleID"] == sample_id]
                sample_df_known = df_known[df_known["sampleID"] == sample_id]
                sample_gdf_known = self._get_known_pred_locations(sample_df_known)
                self._plot_single_sample_boot_error(
                    sample_gdf, sample_id, basemap, sample_gdf_known=sample_gdf_known
                )

    def _get_known_pred_locations(self, df_known):
        return gpd.GeoDataFrame(
            df_known, geometry=gpd.points_from_xy(df_known.x, df_known.y)
        )
