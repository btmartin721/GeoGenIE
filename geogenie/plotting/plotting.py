import logging
import os
import traceback
import wget
from pathlib import Path

import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from optuna import visualization
from shapely.geometry import Point
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from geogenie.utils.scorers import haversine


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
        """Plot training and validation loss."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label="Training Loss")
        plt.plot(val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss Over Epochs", fontsize=self.fontsize)
        plt.xlabel("Epochs", fontsize=self.fontsize)
        plt.ylabel("Loss", fontsize=self.fontsize)
        plt.ylim(
            bottom=min(np.min(val_loss), 0),
            top=min(np.max(val_loss), np.min(val_loss) * 5),
        )
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
        output_dir,
        buffer=0.1,
        show=False,
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
            length_scale=1.0, length_scale_bounds=(1e-2, 1e5)
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
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
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(actual_coords[:, 0], actual_coords[:, 1])
        )

        outshp = os.path.join(output_dir, "shapefile")

        # Download and save map data.
        wget.download(url, outshp)
        mapfile = os.path.join(outshp, url.split("/")[-1])

        try:
            mapdata = gpd.read_file(os.path.join(outshp, url.split("/")[-1]))
        except Exception:
            self.logger.error(f"Could not read map file {mapfile} from url {url}.")
            raise
        ax.set_xlim(
            [actual_coords[:, 0].min() - buffer, actual_coords[:, 0].max() + buffer]
        )
        ax.set_ylim(
            [actual_coords[:, 1].min() - buffer, actual_coords[:, 1].max() + buffer]
        )
        mapdata[mapdata.geometry.intersects(gdf.unary_union.envelope)].boundary.plot(
            ax=ax,
            edgecolor="k",
            linewidth=5,
            facecolor="none",
            label="Haversine Error",
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

        if show:
            plt.show()
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_pca_curve(self, x, vr, knee, output_dir, prefix, show=False):
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

        outfile = os.path.join(output_dir, "plots", f"{prefix}_pca_curve.png")

        if show:
            plt.show()
        plt.savefig(outfile, facecolor="white", bbox_inches="tight")

    def plot_dbscan_clusters(
        self,
        Xy,
        output_dir,
        prefix,
        dataset,
        labels,
        show=False,
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
        # Load USA states data
        usa_states = gpd.read_file(
            "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_20m.zip"
        )

        # Extract Arkansas
        arkansas = usa_states[usa_states["STUSPS"] == "AR"]

        # Plotting
        arkansas.plot(ax=ax, color="white", edgecolor="black")

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
            output_dir, "plots", f"{prefix}_outlier_clustering_{dataset}.png"
        )

        if show:
            plt.show()
        fig.savefig(outfile, facecolor="white", bbox_inches="tight")
        plt.close()
