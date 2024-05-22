import json
import logging
import os
import tempfile
import warnings
import zipfile
from pathlib import Path

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
from kneed import KneeLocator
from optuna import exceptions as optuna_exceptions
from optuna import visualization
from scipy.stats import gamma
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from geogenie.samplers.samplers import GeographicDensitySampler, custom_gpr_optimizer
from geogenie.utils.exceptions import TimeoutException
from geogenie.utils.spatial_data_processors import SpatialDataProcessor
from geogenie.utils.utils import time_limit

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
        url,
        show_plots=False,
        fontsize=18,
        filetype="png",
        dpi=300,
        remove_splines=False,
    ):
        """
        A class dedicated to generating and managing plots for data visualization, particularly in the context of geographical and statistical data analysis. It provides functionalities to customize plot appearance and supports saving plots to specified output directories.

        Args:
            device (str): The device used for plotting, typically 'cpu' or 'cuda'.
            output_dir (str): The directory where plots will be saved.
            prefix (str): A prefix added to the names of saved plot files.
            basemap_fips (str): FIPS code for base map.
            basemap_highlights (list): List of counties to highlight gray on base map.
            url (str): URL from which to download basemap.
            show_plots (bool, optional): Flag to determine if plots should be displayed inline. Defaults to False.
            fontsize (int, optional): Font size used in plots. Defaults to 18.
            filetype (str, optional): File type/format for saving plots. Defaults to 'png'.
            dpi (int, optional): Dots per inch, specifying the resolution of plots. Defaults to 300.
            remove_splines (bool, optional): If True, then remove splines from map plots. If False, then only top and right splines are included. Defaults to False.

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
        self.remove_splines = remove_splines

        self.outbasepath = Path(
            self.output_dir, "plots", f"{self.prefix}.{self.filetype}"
        )

        self.logger = logging.getLogger(__name__)

        # Initialize SpatialDataProcessor instance
        self.processor = SpatialDataProcessor(
            output_dir=self.outbasepath.parent,
            basemap_fips=self.basemap_fips,
            logger=self.logger,
        )

        self.basemap = self.processor.extract_basemap_path_url(url)

        if "STATEFP" not in self.basemap.columns:
            self.logger.warning(
                "'shapefile does not contain 'STATEFP' columns. cannot subset shapefile by FIPS code."
            )

        if not isinstance(self.basemap_fips, str):
            self.basemap_fips = str(self.basemap_fips)

        if len(self.basemap_fips) == 1:
            self.basemap_fips = "0" + self.basemap_fips

        self.basemap = self.basemap[self.basemap["STATEFP"] == self.basemap_fips]

        keep_spines = False if remove_splines else True

        # Adjust matplotlib settings globally.
        sizes = {
            "axes.labelsize": self.fontsize,
            "axes.titlesize": self.fontsize,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.top": False,
            "ytick.right": False,
            "figure.titlesize": self.fontsize,
            "figure.labelsize": self.fontsize,
            "xtick.labelsize": self.fontsize,
            "ytick.labelsize": self.fontsize,
            "font.size": self.fontsize,
            "legend.fontsize": self.fontsize,
            "legend.title_fontsize": self.fontsize,
            "legend.frameon": False,
            "legend.markerscale": 2.0,
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

        fn = Path(filename).with_suffix(self.filetype)
        outfile = self.outbasepath.with_name(fn)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_smote_bins(self, df, bins, df_orig, bins_orig):
        """
        Plots scatter plots before and after SMOTE (Synthetic Minority Over-sampling Technique) to visualize the effect of oversampling on the data distribution. The method creates a subplot with two scatter plots: one showing the original data and the other showing the data after SMOTE has been applied.

        Args:
            df (pandas DataFrame): DataFrame containing the data after SMOTE oversampling.
            bins (array-like): Array of bin labels for the data after SMOTE.
            df_orig (pandas DataFrame): DataFrame containing the original data before SMOTE.
            bins_orig (array-like): Array of original bin labels before SMOTE.

        Notes:
            - This function visually compares the geographical distribution of data before and after SMOTE.
            - Each plot shows data points colored by their bin labels, providing insight into the oversampling process.
        """
        fig, axs = plt.subplots(1, 2, figsize=(24, 12))

        ax = self._plot_smote_scatter(
            df_orig, bins_orig, axs[0], "Before Simulations", "upper right"
        )
        ax2 = self._plot_smote_scatter(
            df, bins, axs[1], "After Simulations", "upper left"
        )

        ax = self._remove_spines(ax)
        ax2 = self._remove_spines(ax2)

        plt.subplots_adjust(wspace=0.25)

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_oversampling_scatter.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def _remove_spines(self, ax):
        if self.remove_splines:
            try:
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.xaxis.set_ticks_position("none")
                ax.yaxis.set_ticks_position("none")
                ax.set_xticks([])
                ax.set_xticks([], minor=True)
                ax.set_yticks([])
                ax.set_yticks([], minor=True)
                plt.axis("off")
            except AttributeError:
                plt.axis("off")
        return ax

    def _plot_smote_scatter(self, df, bins, ax, title, loc):
        """
        Creates a scatter plot for visualizing data points with their associated bin labels.

        This method is used internally by `plot_smote_bins` to generate individual scatter plots.

        Args:
            df (pandas DataFrame): DataFrame containing the data to be plotted.
            bins (array-like): Array of bin labels for the data.
            ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.
            title (str): Title of the scatter plot.
            loc (str): Location of the legend in the plot.

        Notes:
            - This function is a helper method and is not intended to be called directly.
            - It adds a scatter plot to the provided Axes object, with data points colored by their bin labels.
        """
        df = df.copy()
        df["bins"] = bins
        n_bins = len(df["bins"].unique())

        try:
            df["size"] = df["sample_weights"].astype(int)
        except (KeyError, TypeError):
            df["size"] = 1

        # Ensures correct CRS.
        gdf = self.processor.to_geopandas(df)

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        ax = sns.scatterplot(
            data=gdf,
            x=gdf.geometry.x,
            y=gdf.geometry.y,
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

        # extract the existing handles and labels
        h, l = ax.get_legend_handles_labels()

        # slice the appropriate section of l and h to include in the legend
        ax.legend(h[n_bins:], l[n_bins:], bbox_to_anchor=anchor, loc=loc)

        return ax

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

            fn = f"{self.prefix}_opt_history.{self.filetype}"
            outfile = self.outbasepath.with_name(fn)
            hist_fig.write_image(str(outfile))

        except Exception as e:
            self.logger.error(f"Could not create Optuna plot: {e}")
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

        fn = f"{self.prefix}_summary_stats_selected_metrics.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)

        if self.show_plots:
            plt.show()

        # Save the plot
        plt.savefig(outfile, facecolor="white", bbox_inches="tight", dpi=300)
        plt.close()
        self.logger.info(f"Facet grid plot saved to {outfile}")

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
        ax[0].legend_.set_title("Metric")

        sns.move_legend(
            ax[0],
            loc="center",
            bbox_to_anchor=(0.5, 1.45),
            ncol=len(df_melt_errordist["variable"].unique()),
        )

        ax[1].set_title("Distribution of Bootstrapped Error")
        ax[1].set_xlabel("Correlation Coefficient")
        ax[1].set_ylabel("Density")
        sns.move_legend(
            ax[1],
            loc="center",
            bbox_to_anchor=(0.5, 1.45),
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
        sns.move_legend(ax[0], loc="center", bbox_to_anchor=(0.5, 1.15))

        ax[1].set_title("Bootstrapped Prediction Error")
        ax[1].set_xlabel("Metric")
        ax[1].set_ylabel("Correlation Coefficient")
        sns.move_legend(ax[1], loc="center", bbox_to_anchor=(0.5, 1.15), ncol=2)
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
            "percentiles_75": "75th Percentile of Error",  # typo in original data
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

    def plot_scatter_samples_map(self, y_true_train, y_true, dataset):
        """
        Plots geographical scatter plots of training and test/validation sample densities. This method creates a subplot with two scatter plots, one showing the density of training samples and the other for test or validation samples.

        Args:
            y_true_train (np.array): Array of actual geographical coordinates for the training dataset.
            y_true (np.array): Array of actual geographical coordinates for the test or validation dataset.
            dataset (str): Specifies whether the dataset is 'test' or 'validation'.

        Notes:
            - The method visualizes the geographical distribution of training and test/validation samples.
            - It uses scatter plots to represent the density of samples in different geographical areas.
            - The scatter plots are overlaid on top of a base map obtained from the specified shapefile URL.
        """
        gdf_actual_train = self.processor.to_geopandas(y_true_train)
        gdf_actual_val_test = self.processor.to_geopandas(y_true)

        fig, ax = plt.subplots(1, 2, figsize=(12, 12))

        # Plot KDE as contour on the maps
        for i, title, gdf_y in zip(
            range(2),
            ["Training Sample Density", f"{dataset.capitalize()} Sample Density"],
            [gdf_actual_train, gdf_actual_val_test],
        ):
            # Plot the basemap
            ax[i] = self.basemap.plot(
                ax=ax[i],
                color="none",
                edgecolor="k",
                linewidth=3,
                facecolor="none",
                label="State/ County Lines",
            )
            ax[i] = sns.scatterplot(
                x=gdf_y.geometry.x,
                y=gdf_y.geometry.y,
                s=plt.rcParams["lines.markersize"] ** 2 * 4,
                c="darkorchid",
                alpha=0.6,
                ax=ax[i],
            )

            ax[i] = self._remove_spines(ax[i])

            gray_gdf = self._highlight_counties(
                self.basemap_highlights, self.basemap, ax=ax[i]
            )

            ax[i].set_title(title)
            ax[i].set_xlabel("Longitude")
            ax[i].set_ylabel("Latitude")
            ax[i].set_aspect("equal", "box")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_train_{dataset}_sample_densities.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_geographic_error_distribution(
        self,
        actual_coords,
        predicted_coords,
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
            centroids (np.array or geopandas.GeoDataFrame, optional): Array of centroids to be plotted. Defaults to None.

        Notes:
            - This method produces two subplots: one showing the spatial distribution of prediction errors and the others showing the uncertainty of these predictions.
        """
        gdf_actual = self.processor.to_geopandas(actual_coords)
        gdf_pred = self.processor.to_geopandas(predicted_coords)
        xmin, ymin, xmax, ymax = self.processor.calculate_bounding_box(gdf_actual)

        if centroids is not None:
            gdf_centroids = self.processor.to_geopandas(centroids)

        actual_coords = self.processor.to_numpy(gdf_actual)
        predicted_coords = self.processor.to_numpy(gdf_pred)

        haversine_errors = self.processor.haversine_distance(
            actual_coords, predicted_coords
        )

        # Introduce a little random noise into the coordinates.
        sigma = 0.005
        for i in range(2):
            actual_coords[:, i] = np.random.normal(actual_coords[:, i], sigma)

        # Run Gaussian Process Regression with RBF kernel
        gp = self._run_gpr(actual_coords, haversine_errors, n_restarts_optimizer=50)

        # Create a grid over the area of interest
        grid_x, grid_y = np.meshgrid(
            np.linspace(xmin - buffer, xmax + buffer, 1000),
            np.linspace(ymin - buffer, ymax + buffer, 1000),
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

        ax = self._remove_spines(ax)
        ax2 = self._remove_spines(ax2)

        # Customization.

        if not self.remove_splines:
            ax.set_xlabel("Latitude")
            ax.set_ylabel("Longitude")
            ax2.set_xlabel("Latitude")
            ax2.set_ylabel("Longitude")

        # Ensure correct scale and aspect ratio
        ax.set_aspect("equal", "box")
        ax2.set_aspect("equal", "box")

        def roundup(x):
            x -= x % -100
            return x

        if (
            roundup(np.max(haversine_errors)) > max_colorscale - 100
            and max_colorscale > 100
        ):
            # Round to nearest 100.
            vmax = min(roundup(np.max(haversine_errors)), max_colorscale)
        else:
            vmax = max(roundup(np.max(haversine_errors)), max_colorscale)
        vmax_std = min(roundup(np.max(error_std)), 100)

        # Define colormap and normalization
        cmap = plt.get_cmap("magma_r")
        cmap.set_bad(cmap(0))  # Set color for NaN values
        uncertainty_cmap = plt.get_cmap("magma_r")
        uncertainty_cmap.set_bad(uncertainty_cmap(0))  # Set color for NaN values

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
            extend="both",  # Include values beyond the defined range
        )

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
            extend="both",  # Include values beyond the defined range
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

        # TODO: Figure out how to make these quiver arrows look better.
        # Error Vector Visualization (Quiver plot)
        # Calculate error vectors
        # error_vectors = predicted_coords - actual_coords

        # # Calculate the lengths and angles of the error vectors
        # lengths = np.linalg.norm(error_vectors, axis=1)
        # angles = np.arctan2(error_vectors[:, 1], error_vectors[:, 0])

        # ax.quiver(
        #     gdf_actual.x.to_numpy(),
        #     gdf_actual.y.to_numpy(),
        #     error_vectors[:, 0],
        #     error_vectors[:, 1],
        #     angles,
        #     angles="xy",
        #     scale_units="xy",
        #     scale=1,
        #     color="red",
        #     label="Prediction Error",
        # )

        for i, a in enumerate([ax, ax2]):
            # Plot the basemap
            a = self.basemap.plot(
                ax=a,
                color="none",
                edgecolor="k",
                linewidth=3,
                facecolor="none",
                label="State/ County Lines",
            )

            if i == 0:
                ax = a
            else:
                ax2 = a

        if dataset.lower() == "val":
            dataset = "validation"

        # Plot centroids layer.
        if centroids is not None:
            ax = self._plot_scatter_map(
                dataset,
                ax,
                self.processor.to_numpy(gdf_centroids),
                marker_scale_factor,
                mult_factor=1.4,
                label="Centroids",
                color="orange",
            )

        # Plot samples layer
        ax = self._plot_scatter_map(dataset, ax, actual_coords, marker_scale_factor)
        ax2 = self._plot_scatter_map(dataset, ax2, actual_coords, marker_scale_factor)

        ncol = 2 if centroids is not None else 1

        # Add legend
        ax.legend(bbox_to_anchor=(0.5, 1.7), loc="upper center", ncol=ncol)

        cbar = self._set_cbar_fontsize(cbar)
        uncert_cbar = self._set_cbar_fontsize(uncert_cbar)

        cbar.ax.set_title("Prediction Error (km)\n", fontsize=self.fontsize)
        uncert_cbar.ax.set_title("Interpolation Uncertainty\n", fontsize=self.fontsize)

        plt.subplots_adjust(wspace=0.5, hspace=0.05)

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_geographic_error_{dataset}.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

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
        color="blue",
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
        plt.scatter(
            coords[:, 0],
            coords[:, 1],
            s=plt.rcParams["lines.markersize"] ** exp_factor * mult_factor,
            c=color,
            alpha=alpha,
            label=f"{label} ({dataset.capitalize()} Set)",
        )
        return plt.gca()

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

    def plot_cumulative_error_distribution(self, data, fn, percentiles, median, mean):
        """
        Generate an ECDF plot for the given data.

        Args:
            data (array-like): The dataset for which the ECDF is to be plotted. Should be a 1-D array of prediction errors.
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

        cmap = plt.colormaps.get_cmap("magma_r")

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
        plt.legend(loc="upper left", bbox_to_anchor=(1.04, 1), borderpad=1)

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

    def plot_zscores(self, z, fn):
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

        cmap = plt.colormaps.get_cmap("magma_r")
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        # Histogram (Density Plot with Gradient)
        plt.subplot(1, 3, 1)

        # Compute KDE
        kde = stats.gaussian_kde(errors)
        x_values = np.linspace(np.min(errors), np.max(errors), 150)
        kde_values = kde(x_values)

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

        fn = Path(outfile).with_suffix("." + self.filetype)
        outfile = self.outbasepath.with_name(fn.name)
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def polynomial_regression_plot(
        self,
        actual_coords,
        predicted_coords,
        dataset,
        degree=3,
        dtype=torch.float32,
        max_ylim=None,
        max_xlim=None,
        n_xticks=5,
    ):
        """
        Creates a polynomial regression plot with the specified degree.

        Args:
            actual_coords (np.array): Array of actual geographical coordinates.
            predicted_coords (np.array): Array of predicted geographical coordinates by the model.
            dataset (str): Specifies the dataset being used, should be either 'test' or 'validation'.
            degree (int): Polynomial degree to fit. Defaults to 3.
            dtype (torch.dtype): PyTorch data type to use. Defaults to torch.float32.
            max_ylim (int): Maximum y-axis (prediction error) value to plot. Defaults to None (don't adjust y-axis limits).
            max_xlim (float): Maximum X-axis (sample density) value to plot. Defaults to None (don't adjust x-axis limits).
            n_xticks (int): Number of major X-axis ticks to use. Only applied if max_xlim is not None. Defaults to 4.

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

        gdf_actual = self.processor.to_geopandas(actual_coords)
        gdf_pred = self.processor.to_geopandas(predicted_coords)

        # Project to UTM zone (adjust the zone number as per the location)
        # Arkansas is generally in UTM zone 15N
        gdf_basemap = self.basemap.copy()
        gdf_basemap = gdf_basemap.to_crs(epsg=32615)

        # Calculate area in square meters and convert to square kilometers
        gdf_basemap["Area_km2"] = (
            gdf_basemap["geometry"].area / 1e6
        )  # convert sq meters to sq km

        sampler = GeographicDensitySampler(
            self.processor.to_pandas(gdf_actual),
            use_kde=True,
            use_kmeans=False,
            max_clusters=10,
            max_neighbors=50,
            verbose=0,
            dtype=dtype,
        )

        x = sampler.density

        # Calculate Haversine error for each pair of points
        y = self.processor.haversine_distance(gdf_actual, gdf_pred)

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

        ax = sns.regplot(
            x=xfit,
            y=yfit,
            fit_reg=True,
            n_boot=1000,
            line_kws={"lw": 5, "color": "darkorchid"},
            label=f"Pearson's R = {r:.2f}\nP-value {p}",
        )

        kneedle = KneeLocator(
            xfit,
            yfit,
            curve="convex",
            direction="decreasing",
            interp_method="polynomial",
            polynomial_degree=3,
        )

        opt_samp_density = kneedle.knee
        self.logger.info(f"Optimal sampling density: {opt_samp_density:.2f}")

        # Calculate total area
        total_area_km2 = gdf_basemap["Area_km2"].sum()

        # Calculate proportional sampling density
        proportional_density = len(actual_coords) / total_area_km2

        gdf_basemap["Required_Samples"] = (
            gdf_basemap["Area_km2"] * proportional_density
        ).astype(int)

        def calculate_95_ci(dataframe, column_name):
            """
            Calculate the 95% confidence interval for a given column in a pandas DataFrame.

            Args:
                dataframe (pd.DataFrame): The input DataFrame.
                column_name (str): The name of the column for which to calculate the 95% CI.

            Returns:
                tuple: A tuple containing the lower and upper bounds of the 95% CI.
            """
            if column_name not in dataframe:
                msg = f"Column {column_name} not found in DataFrame."
                self.logger.error(msg)
                raise ValueError(msg)

            data = dataframe[column_name].dropna()  # Drop missing values
            mean = data.mean()
            sem = stats.sem(data)  # Standard error of the mean
            margin_of_error = sem * stats.t.ppf(
                (1 + 0.95) / 2, len(data) - 1
            )  # 95% confidence level

            lower_bound = mean - margin_of_error
            upper_bound = mean + margin_of_error

            return lower_bound, upper_bound

        mean_samples_per_county = gdf_basemap["Required_Samples"].mean()
        lower_95ci, upper_95ci = calculate_95_ci(gdf_basemap, "Required_Samples")

        ds = "validation" if dataset == "val" else dataset

        # Plotting the results
        ax.scatter(
            x,
            y,
            alpha=0.7,
            color="lightseagreen",
            label=f"{ds.capitalize()} Set Samples",
            s=plt.rcParams["lines.markersize"] ** 2 * 5,
            lw=2,
            edgecolors="k",
        )

        ax.axvline(
            opt_samp_density,
            label=f"Optimal Sample Density = {opt_samp_density:.2f}"
            "\n"
            f"Mean Samples / County: {round(mean_samples_per_county, 2)}"
            "\n"
            f"95% CI: {max(round(lower_95ci, 2), 0)}, {round(upper_95ci, 2)}",
            color="orange",
            linewidth=2,
            linestyle="dashed",
        )

        def roundup(x):
            x -= x % -100
            return x

        if max_ylim is not None:
            if roundup(np.max(y)) > max_ylim - 100 and max_ylim > 100:
                # Round to nearest 100.
                ymax = min(roundup(np.max(y)), max_ylim)
            else:
                ymax = max(roundup(np.max(y)), max_ylim)
            ax.set_ylim([0, ymax])

        if max_xlim is not None:
            if not isinstance(max_xlim, float):
                try:
                    max_xlim = float(max_xlim)
                except Exception:
                    msg = f"max_xlim could not be coerced to type float. Value must be numeric, but got: {type(max_xlim)}"
                    self.logger.error(msg)
                    raise TypeError(msg)
            ax.set_xlim([0.0, max_xlim])
            ax.set_xticks(np.linspace(0.0, max_xlim, num=n_xticks, endpoint=True))

        ax.set_ylabel("Prediction Error (km)")
        ax.set_xlabel(r"Sample Density (Samples / $km^2$)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.4, 0.9))

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

    def plot_dbscan_clusters(self, Xy, dataset, labels):
        """
        Plots the clusters formed by DBSCAN algorithm on geographical data. Each cluster is visualized with a different color, and outliers are marked distinctly.

        Args:
            Xy (np.array): Array containing the data with pre-transformed 'x' and 'y' coordinates.
            dataset (str): Name of the dataset being used.
            labels (np.array): Cluster labels assigned by DBSCAN to each data point.

        Notes:
            - The function converts the data to a GeoDataFrame for geographical plotting.
            - Different clusters are visualized in different colors, with outliers typically in red.
        """
        # Convert to GeoDataFrame for plotting
        gdf = self.processor.to_geopandas(Xy)
        gdf["cluster"] = labels

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

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

    def plot_geographical_heatmap(self, data, weights, title="Sampling Weight Heatmap"):
        """
        Plots a geographical heatmap representing the sampling weights of data points. The heatmap provides a visual representation of the density or weight of data points in geographical space.

        Args:
            data (pandas DataFrame): DataFrame containing 'longitude' and 'latitude' columns for geographical data points.
            weights (np.ndarray): Array of weights corresponding to each data point in the DataFrame.
            title (str, optional): Title for the heatmap plot. Defaults to "Sampling Weight Heatmap".

        Notes:
            - The function creates a GeoDataFrame from the provided data and weights for plotting.
            - The heatmap uses color intensity to represent the density or weight of data points in different geographical locations.
        """

        gdf = self.processor.to_geopandas(data)
        gdf["weights"] = weights

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        gdf.plot(
            column="weights",
            ax=ax,
            legend=True,
            cmap="viridis",
            markersize=20,
        )

        gray_gdf = self._highlight_counties(
            self.basemap_highlights, self.basemap, ax=ax
        )

        ax = self._remove_spines(ax)

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
        self, data, weights, marker_scale_factor=3, title="Geographic Scatter Plot"
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

        # Ensure correct CRS.
        gdf = self.processor.to_geopandas(data)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        ax = sns.scatterplot(
            x=gdf.geometry.x,
            y=gdf.geometry.y,
            size=weights,
            sizes=(
                plt.rcParams["lines.markersize"] ** marker_scale_factor * 4,
                plt.rcParams["lines.markersize"] ** marker_scale_factor * 16,
            ),
            alpha=0.5,
            c="darkorchid",
            ax=ax,
        )

        ax = self._remove_spines(ax)

        gray_gdf = self._highlight_counties(
            self.basemap_highlights, self.basemap, ax=ax
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

    def plot_outliers(self, mask, y_true):
        """
        Plots a scatter plot to visualize the identified outliers in the dataset. Outliers are marked distinctly to
        differentiate them from the regular data points.

        Args:
            mask (np.array): A boolean array where 'True' indicates an outlier.
            y_true (np.array): Array of actual coordinates.

        Notes:
            - The function visualizes outliers on a geographical map, aiding in the identification of anomalous data points.
        """
        # Ensure correct CRS.
        gdf = self.processor.to_geopandas(y_true)
        df = self.processor.to_pandas(gdf)

        df["Outliers"] = ~mask
        df["Outliers"] = df["Outliers"].astype(str)
        df = df[~df["x"].isna()]
        df["Sizes"] = "Non-Outlier"
        df.loc[df["Outliers"] == "True", "Sizes"] = "Outlier"

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        if self.basemap_highlights is not None:
            gray_gdf = self._highlight_counties(
                self.basemap_highlights, self.basemap, ax=ax
            )

        ax = sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="Outliers",
            size="Sizes",
            sizes=(1000, 100),
            size_order=["Non-Outlier", "Outlier"],
            palette="Set2",
            alpha=0.7,
            ax=ax,
        )

        ax = self._remove_spines(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Outliers Removed from Dataset")

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[3:],
            labels[3:],
            loc="upper left",
            bbox_to_anchor=(1.04, 1),
            markerscale=1,
        )

        ax.legend_.set_title("Outlier Status")

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
        """
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        for ax, data_type in zip(axs, ["geographic", "genetic"]):
            data = genetic_data if data_type == "genetic" else geographic_data
            if data_type == "geographic":
                # Plot the basemap
                ax = self.basemap.plot(
                    ax=ax,
                    color="none",
                    edgecolor="k",
                    linewidth=3,
                    facecolor="none",
                    label="State/ County Lines",
                )
            # Ensure correct CRS.
            gdf = self.processor.to_geopandas(data)
            data = self.processor.to_numpy(gdf)

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

        pth = Path(filename).with_suffix(f".{self.filetype}")
        outfile = self.outbasepath.with_name(str(pth.name))
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")
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

    def plot_sample_with_density(
        self, df, sample_id, df_known=None, dataset=None, gray_counties=None
    ):
        if dataset is None:
            msg = "dataset argument cannot be NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

        if isinstance(df, gpd.GeoDataFrame):
            gdf = df.copy()
        else:
            gdf = self.processor.to_geopandas(df)

        if df_known is not None:
            gdf_known = self.processor.to_geopandas(df_known)

        xmin, ymin, xmax, ymax = self.processor.calculate_bounding_box(gdf)

        # Calculate density contours using KDE
        coords = self.processor.to_numpy(gdf)
        kde = stats.gaussian_kde(coords.T)
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        grid = np.vstack([xx.ravel(), yy.ravel()])
        zz = kde(grid).reshape(xx.shape)

        # Determine density levels
        z_sorted = np.sort(zz.ravel())
        cumulative_z = np.cumsum(z_sorted)
        total_z = cumulative_z[-1]
        level_50 = z_sorted[np.searchsorted(cumulative_z, 0.50 * total_z)]
        level_70 = z_sorted[np.searchsorted(cumulative_z, 0.70 * total_z)]
        level_90 = z_sorted[np.searchsorted(cumulative_z, 0.90 * total_z)]
        levels = np.sort([level_50, level_70, level_90])

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot density contours
        colors = ["#E69F00", "#56B4E9", "#CC79A7"]
        ax.contour(xx, yy, zz, levels=levels, colors=colors)

        # Create custom legend for the contours
        contour_lines = [mlines.Line2D([0], [0], color=color, lw=2) for color in colors]

        grays = [mpatches.Patch(facecolor="darkgrey", edgecolor="k")]

        gdf = gdf.clip(self.basemap)

        if df_known is not None:
            gdf_known = gdf_known.clip(self.basemap)

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        if gray_counties is not None:
            gdf_gray = self._highlight_counties(gray_counties, self.basemap, ax)

        labels = ["90% Density", "70% Density", "50% Density"]

        # Plot bootstrapped points with reduced opacity
        ax.scatter(
            gdf.geometry.x,
            gdf.geometry.y,
            alpha=0.4,
            color="gray",
            label="Predicted Points",
        )

        ax = self._remove_spines(ax)

        # Plot mean point
        mean_lat, mean_lon = gdf.dissolve().centroid.y, gdf.dissolve().centroid.x
        ax.scatter(
            mean_lon, mean_lat, s=200, color="k", marker="X", label="Predicted Locality"
        )

        # Plot known points
        if df_known is not None:
            if df_known.shape[0] == 1:
                mean_known_lat, mean_known_lon = (
                    gdf_known.geometry.y,
                    gdf_known.geometry.x,
                )
            else:
                mean_known_lat, mean_known_lon = (
                    gdf_known.dissolve().centroid.y,
                    gdf_known.dissolve().centroid.x,
                )

            ax.scatter(
                mean_known_lon,
                mean_known_lat,
                s=200,
                color="k",
                marker="^",
                label="Recorded Locality",
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{sample_id}: Bootstrapped Locality Predictions")

        handles, labs = ax.get_legend_handles_labels()
        contour_lines += handles + grays
        labels += labs + ["CWD Mgmt Zone"]

        ax.legend(
            contour_lines, labels, loc="center", bbox_to_anchor=(0.5, 1.3), ncol=2
        )

        ax.set_aspect("equal", "box")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_bootstrap_ci_plot_{dataset}_{sample_id}.{self.filetype}"
        outdir = self.outbasepath.parent / "bootstrapped_sample_ci"
        outfile = outdir / fn
        outdir.mkdir(exist_ok=True, parents=True)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def _highlight_counties(self, gray_counties, gdf, ax=None):
        if gray_counties is not None:

            if isinstance(gray_counties, str):
                gray_counties = gray_counties.split(",")

            if "NAME" not in gdf.columns:
                self.logger.warning(
                    f"Cannot highlight basemap counties. Attribute 'Name' not in provided basemap shapefile: {self.url}"
                )
            # Filtering the counties to be colored gray
            gray_county_gdf = gdf[gdf["NAME"].isin(gray_counties)]
            gray_county_gdf.plot(ax=ax, color="darkgray", edgecolor="k", alpha=0.5)
            return gray_county_gdf
        return gdf

    def visualize_oversample_clusters(self, arr, cluster_labels, sample_origin_list):
        """
        Visualize the genotypes and their clusters in a 2D scatter plot.

        Args:
            arr (np.ndarray): Array to use for clustering.
            cluster_labels (np.ndarray: Cluster labels to use.
            sample_origin_list: (list): List of sample origins (synthetic versus original).
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        cluster_sizes = np.ones_like(cluster_labels)
        n_bins = len(np.unique(cluster_labels))

        mask = sample_origin_list == "synthetic"
        cluster_sizes[mask] *= 100
        cluster_sizes[~mask] *= 50

        # Get a geopandas GeoDataFrame.
        gdf = self.processor.to_geopandas(arr)

        # Plot the basemap
        ax = self.basemap.plot(
            ax=ax,
            color="none",
            edgecolor="k",
            linewidth=3,
            facecolor="none",
            label="State/ County Lines",
        )

        if self.basemap_highlights is not None:
            gray_gdf = self._highlight_counties(
                self.basemap_highlights, self.basemap, ax=ax
            )

        ax = sns.scatterplot(
            x=gdf.geometry.x,
            y=gdf.geometry.y,
            hue=cluster_labels,
            style=sample_origin_list,
            style_order=["original", "synthetic"],
            palette="Set2",
            s=150,
            linewidth=1,
            edgecolor="k",
            ax=ax,
        )

        ax = self._remove_spines(ax)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Oversampling Groups")

        h, l = ax.get_legend_handles_labels()

        # Exclude first n_bins labels.
        ax.legend(
            h[n_bins:],
            l[n_bins:],
            loc="upper left",
            bbox_to_anchor=(1.04, 1.0),
        )

        ax.legend_.set_title("Sample Origin")

        if self.show_plots:
            plt.show()

        fn = f"{self.prefix}_coords_train_clusters_oversampled.{self.filetype}"
        outfile = self.outbasepath.with_name(fn)
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    @property
    def pfx(self):
        return self.prefix

    @pfx.setter
    def pfx(self, value):
        self.prefix = value

    @property
    def outdir(self):
        return self.prefix

    @outdir.setter
    def outdir(self, value):
        self.output_dir = value

    @property
    def obp(self):
        return self.outbasepath

    @obp.setter
    def obp(self, value):
        self.outbasepath = value
