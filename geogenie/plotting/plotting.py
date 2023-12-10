import logging
import os
import traceback

import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wget
from optuna import visualization
from scipy.stats import gamma
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from geogenie.utils.scorers import haversine
from geogenie.utils.utils import (
    rmse_to_distance,
    custom_gpr_optimizer,
    geo_coords_is_valid,
)


class PlotGenIE:
    def __init__(self, device, output_dir, prefix, show_plots=False, fontsize=18):
        self.device = device
        self.output_dir = output_dir
        self.prefix = prefix
        self.show_plots = show_plots
        self.fontsize = fontsize

        self.logger = logging.getLogger(__name__)

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
        plt.xlabel("Bootstrap Replicate", fontsize=self.fontsize)
        plt.ylabel("Duration (s)", fontsize=self.fontsize)
        plt.title(
            "Rolling Average Time of per-Bootstrap Model Training",
            fontsize=self.fontsize,
        )
        plt.legend()
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

        if self.show_plots:
            plt.show()
        plt.close()

    def plot_history(self, train_loss, val_loss, filename):
        """Automatically plot training and validation loss with appropriate scaling.

        Args:
            train_loss (list): List of training loss values.
            val_loss (list): List of validation loss values.
            filename (str): Name of the file to save the plot.

        """
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs", fontsize=self.fontsize)
        plt.xlabel("Epochs", fontsize=self.fontsize)

        # Determine the scaling based on loss values
        max_loss = max(np.max(train_loss), np.max(val_loss))
        min_loss = min(np.min(train_loss), np.min(val_loss))

        if max_loss / min_loss > 10:  # Threshold for log scale
            plt.yscale("log")
            plt.ylabel("Log Loss", fontsize=self.fontsize)
            plt.ylim(bottom=max(min_loss, 1e-5))  # Avoid log(0) error
        else:
            plt.ylabel("Loss", fontsize=self.fontsize)

        plt.legend()
        plt.savefig(filename, facecolor="white", bbox_inches="tight")

        if self.show_plots:
            plt.show()
        plt.close()

    def make_optuna_plots(self, study):
        """Visualize Optuna search."""
        if not visualization.is_available():
            self.logger.warn(
                "Could not make plots because plotly and kaleido are not installed."
            )
            return

        try:
            outdir = os.path.join(self.output_dir, "plots")
            importance_fig = visualization.plot_param_importances(study)
            importance_fig.write_image(
                os.path.join(outdir, f"{self.prefix}_param_importances.png")
            )

            edf_fig = visualization.plot_edf(study, target_name="Location Error")
            edf_fig.write_image(os.path.join(outdir, f"{self.prefix}_edf.png"))

            par_fig = visualization.plot_parallel_coordinate(
                study, target_name="Location Error"
            )

            par_fig.write_image(
                os.path.join(outdir, f"{self.prefix}_parallel_coordinates.png")
            )

            slice_fig = visualization.plot_slice(study, target_name="Location Error")
            slice_fig.write_image(os.path.join(outdir, f"{self.prefix}_slices.png"))

            tl_fig = visualization.plot_timeline(study)
            tl_fig.write_image(os.path.join(outdir, f"{self.prefix}_timeline.png"))

            rank_fig = visualization.plot_rank(study, target_name="Location Error")
            rank_fig.write_image(os.path.join(outdir, f"{self.prefix}_rank.png"))

            ctr_fig = visualization.plot_contour(study, target_name="Location Error")
            ctr_fig.write_image(os.path.join(outdir, f"{self.prefix}_contour.png"))

            hist_fig = visualization.plot_optimization_history(
                study, target_name="Location Error"
            )
            hist_fig.write_image(os.path.join(outdir, f"{self.prefix}_opt_history.png"))

        except Exception as e:
            self.logger.error(f"Could not create plot(s): {e}")
            traceback.print_exc()
            raise

    def plot_bootstrap_aggregates(self, df, filename):
        """Make a KDE plot with bootstrap distributions."""
        plt.figure(figsize=(10, 6))

        df_r2 = df[["r2_long", "r2_lat"]]

        df_melt = df_r2.melt()

        sns.kdeplot(
            data=df_melt,
            x="value",
            hue="variable",
            fill=True,
            palette="Set2",
            legend=True,
        )
        plt.title("Distribution of Bootstrapped Error", fontsize=self.fontsize)
        plt.xlabel("Euclidean Distance Error", fontsize=self.fontsize)
        plt.ylabel("Density", fontsize=self.fontsize)
        plt.savefig(filename, facecolor="white", bbox_inches="tight")
        if self.show_plots:
            plt.show()
        plt.close()

    def plot_geographic_error_distribution(
        self,
        actual_coords,
        predicted_coords,
        outfile,
        fontsize,
        url,
        buffer=0.1,
    ):
        # Calculate Haversine error for each pair of points
        haversine_errors = np.array(
            [
                haversine(act[0], act[1], pred[0], pred[1])
                for act, pred in zip(actual_coords, predicted_coords)
            ]
        )

        # Fit Gaussian Process Regressor with a larger initial length scale and
        # no upper bound
        kernel = 1 * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e6)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e6))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=custom_gpr_optimizer,
            n_restarts_optimizer=25,
        )
        gp.fit(actual_coords, haversine_errors)

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

        # Predict error for each point in the grid
        error_predictions = gp.predict(
            np.vstack((grid_x.ravel(), grid_y.ravel())).T
        ).reshape(grid_x.shape)

        # Plot the interpolated errors with corrected color normalization
        fig, ax = plt.subplots(figsize=(12, 12))

        # Correct the normalization range based on the actual error values
        vmin, vmax = np.min(haversine_errors), np.max(haversine_errors)

        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        contour = ax.contourf(
            grid_x, grid_y, error_predictions, levels=1000, cmap="coolwarm", norm=norm
        )
        cbar = plt.colorbar(
            contour,
            ax=ax,
            label="Prediction Error (km)",
            extend="both",
        )

        # Load and plot dynamic boundaries
        self.plot_map(actual_coords, url, self.output_dir, buffer, ax)

        # Create a GeoDataFrame for actual coordinates
        gdf_actual = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in actual_coords], crs="EPSG:4326"
        )

        # Create a GeoDataFrame for predicted coordinates
        gdf_predicted = gpd.GeoDataFrame(
            geometry=[Point(xy) for xy in predicted_coords], crs="EPSG:4326"
        )

        # Transform to WGS 84
        gdf_actual = gdf_actual.to_crs(epsg=4326)
        gdf_predicted = gdf_predicted.to_crs(epsg=4326)

        actual_coords_transformed = np.array(
            [(point.x, point.y) for point in gdf_actual.geometry]
        )

        mms = MinMaxScaler(feature_range=(1000, 10000))

        if len(haversine_errors.shape) > 1:
            msg = f"Invalid shape for haversine_error: {haversine_errors.shape}"
            self.logger.error(msg)
            raise ValueError(msg)
        dist_errors = mms.fit_transform(haversine_errors.reshape(-1, 1))

        # Determine the scaling factor based on plot dimensions
        plot_scaling_factor = (fig.get_size_inches()[0] * fig.get_dpi()) ** 2 / 10

        # Calculate sizes for scatter plot (scaled by error)
        point_sizes = 1 / (dist_errors + 1e-6) * plot_scaling_factor

        # Plotting individual sample points
        ax.scatter(
            actual_coords_transformed[:, 0],
            actual_coords_transformed[:, 1],
            s=point_sizes,
            c="black",
            alpha=0.5,
        )

        # Customization
        ax.set_title(
            "Kriging of Haversine Prediction Errors",
            fontsize=fontsize,
        )
        ax.set_xlabel("Longitude", fontsize=fontsize)
        ax.set_ylabel("Latitude", fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(fontsize)

        cbar.ax.set_ylabel("Prediction Error (km)", fontsize=fontsize)

        if self.show_plots:
            plt.show()
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_map(self, actual_coords, url, output_dir, buffer, ax):
        # Ensure coordinates are valid
        if np.any(np.isnan(actual_coords)) or np.any(np.isinf(actual_coords)):
            self.logger.error("Invalid coordinates detected.")
            raise ValueError("Invalid coordinates in actual_coords.")

        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(actual_coords[:, 0], actual_coords[:, 1]),
            crs="EPSG:4326",
        )

        outshp = os.path.join(output_dir, "shapefile")
        mapfile = os.path.join(outshp, url.split("/")[-1])

        if not os.path.exists(mapfile):
            wget.download(url, outshp, bar=None)

        try:
            mapdata = gpd.read_file(mapfile)
        except Exception:
            self.logger.error(f"Could not read map file {mapfile} from url {url}.")
            raise

        mapdata.crs = "epsg:4326"

        # Set the limits with a buffer
        x_min, x_max = actual_coords[:, 0].min(), actual_coords[:, 0].max()
        y_min, y_max = actual_coords[:, 1].min(), actual_coords[:, 1].max()

        x_min -= buffer
        y_min -= buffer
        x_max += buffer
        y_max += buffer

        cdf = pd.DataFrame(actual_coords, columns=["Longitude", "Latitude"])

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

    def plot_pca_curve(self, x, vr, knee):
        plt.figure()
        plt.plot(x, vr, "-", color="b")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.axvline(
            x=knee,
            label="Selected Number of Components",
            linestyle="--",
            color="orange",
        )
        plt.legend(loc="best")
        plt.title(f"N-Components vs. Explained Variance")

        outfile = os.path.join(
            self.output_dir,
            "plots",
            f"{self.prefix}_pca_curve.png",
        )

        if self.show_plots:
            plt.show()
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_dbscan_clusters(
        self,
        Xy,
        dataset,
        labels,
        url,
        buffer=1.0,
    ):
        """
        Plot DBSCAN clusters for the given DataFrame.

        Args:
            y (numpy.ndarray): Array containing the data with 'x' and 'y' coordinates (pre-transformed).
            eps (float): The maximum distance between two samples for DBSCAN.
            min_samples (int): The number of samples in a neighborhood for DBSCAN.
            show (bool): Whether to show plots in-line.
        """
        # Applying DBSCAN
        df = pd.DataFrame(Xy, columns=["x", "y"] + list(range(Xy.shape[1] - 2)))

        # Convert to GeoDataFrame for plotting
        gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)])
        gdf["cluster"] = labels

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 8))

        ax = self.plot_map(Xy, url, self.output_dir, buffer, ax)

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

        outfile = os.path.join(
            self.output_dir,
            "plots",
            f"{self.prefix}_outlier_clustering_{dataset}.png",
        )

        if self.show_plots:
            plt.show()
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
                ax = self.plot_map(data, url, self.output_dir, buffer, ax)

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
            ax.set_xlabel(xlab, fontsize=self.fontsize)
            ax.set_ylabel(ylab, fontsize=self.fontsize)

        outfile = os.path.join(
            self.output_dir,
            "plots",
            f"{self.prefix}_outliers.png",
        )

        if self.show_plots:
            plt.show()
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, dtype):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ConfusionMatrixDisplay.from_predictions(
            y_true=y_true,
            y_pred=y_pred,
            ax=ax,
        )

        outfile = os.path.join(
            self.output_dir,
            "plots",
            f"{self.prefix}_outlier_{dtype}_confusion_matrix.png",
        )

        if self.show_plots:
            plt.show()
        fig.savefig(outfile, facecolor="white")
        plt.close()

    def plot_gamma_distribution(self, shape, scale, Dg, sig_level, filename, plot_main):
        """
        Plot the gamma distribution.

        Args:
            shape (float): Shape parameter of the gamma distribution.
            scale (float): Scale parameter of the gamma distribution.
            Dg (np.array): Dg statistic for each sample.
            sig_level (float): Significance level.
            filename (str): Name of the file to save the plot.
            plot_main (str): Title of the plot.
        """
        x = np.linspace(0, np.max(Dg), 1000)
        y = gamma.pdf(x, a=shape, scale=scale)

        gamma_threshold = gamma.ppf(1 - sig_level, a=shape, scale=scale)

        plt.figure()
        plt.plot(x, y, color="blue")
        plt.axvline(x=gamma_threshold, color="red", linestyle="--")
        plt.text(
            gamma_threshold,
            plt.ylim()[1],
            f"p = {sig_level}",
            horizontalalignment="right",
        )
        plt.xlabel(f"Gamma(α={shape:.3f}, β={scale:.3f})")
        plt.ylabel("Density")
        plt.title(plot_main)
        plt.grid(True)

        if self.show_plots:
            plt.show()
        plt.savefig(filename, facecolor="white")
        plt.close()
