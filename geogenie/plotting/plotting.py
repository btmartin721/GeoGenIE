import logging
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from optuna import visualization


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
