import csv
import json
import logging
import os
import time
import traceback
from functools import wraps
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
from torch.utils.data import DataLoader

from geogenie.models.models import MLPRegressor
from geogenie.optimize.boostrap import Bootstrap
from geogenie.optimize.optuna_opt import Optimize
from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler, synthetic_resampling
from geogenie.utils.callbacks import EarlyStopping
from geogenie.utils.data import CustomDataset
from geogenie.utils.data_structure import DataStructure
from geogenie.utils.loss import weighted_rmse_loss
from geogenie.utils.scorers import calculate_rmse, haversine_distances_agg, kstest
from geogenie.utils.utils import geo_coords_is_valid, validate_is_numpy

os.environ["TQDM_DISABLE"] = "1"

execution_times = []


def timer(func):
    """
    Decorator that measures and stores the execution time of a function.

    Args:
        func (Callable): The function to be wrapped by the timer.

    Returns:
        Callable: The wrapped function with timing functionality.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append((func.__name__, execution_time))
        return result

    return wrapper


def save_execution_times(filename):
    """
    Appends the execution times to a CSV file. If the file doesn't exist, it creates one.

    Args:
        filename (str): The name of the file where data will be saved.
    """
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        # Check if the file is empty to decide whether to write headers
        f.seek(0, 2)  # Move to the end of the file
        if f.tell() == 0:  # Check if file is empty
            # Write headers only if file is empty
            writer.writerow(["Function Name", "Execution Time"])
        writer.writerows(execution_times)


class GeoGenIE:
    """
    A class designed for predicting geographic localities from genomic SNP (Single Nucleotide Polymorphism) data using neural network and gradient boosting decision tree models.

    GeoGenIE facilitates the integration of genomic data analysis with geographic predictions, aiding in studies like population genetic and molecular ecology.

    Attributes:
        args (argparse.Namespace): Command line arguments used for configuring various aspects of the class.
        genotypes (np.array): Array to store genomic SNP data.
        samples (list): List to store sample identifiers.
        sample_data (pandas.DataFrame): DataFrame to store additional data related to samples.
        locs (np.array): Array to store geographic location data associated with the samples.
        logger (Logger): Logger for logging information and errors.
        device (str): Computational device to be used ('cpu' or 'cuda').
        plotting (PlotGenIE): Instance of the PlotGenIE class for handling data visualization.
        dtype (torch.dtype): torch dtype to use with MLPRegressor.

    Notes:
        - This class is particularly useful in the fields of population genomics, evolutionary biology, and molecular ecology, where geographic predictions based on genomic data are crucial.
        - It requires genomic SNP data as input and utilizes neural network models for making geographic predictions.
    """

    def __init__(self, args):
        """
        Initializes the GeoGenIE class with the provided command line arguments and sets up the necessary environment for geographic predictions from genomic data.

        Args:
            args (argparse.Namespace): Command line arguments containing configurations for data processing, model training, and visualization.

        Notes:
            - The initialization process includes setting up the computational device (CPU or GPU), creating necessary directories for outputs, and initializing the plotting utility.
            - It prepares the class for handling genomic SNP data and associated geographic information.
        """
        self.args = args
        self.genotypes = None
        self.samples = None
        self.sample_data = None
        self.locs = None
        self.dtype = torch.float32 if args.dtype == "float32" else torch.float64

        torch.set_default_dtype(self.dtype)

        # Construct output directory structure to store all output.
        output_dir_list = [
            "plots",
            "training",
            "validation",
            "test",
            "logfiles",
            "predictions",
            "bootstrap",
            "models",
            "optimize",
            "data",
            "shapefile",
            "benchmarking",
        ]

        output_dir = self.args.output_dir
        prefix = self.args.prefix

        [
            Path(os.path.join(output_dir, d)).mkdir(exist_ok=True, parents=True)
            for d in output_dir_list
        ]

        self.logger = logging.getLogger(__name__)

        self.device = "cpu"
        if self.args.gpu_number is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_number)
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )

        self.plotting = PlotGenIE(
            self.device,
            output_dir,
            prefix,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
        )

    def load_data(self):
        """Loads genotypes from VCF file using pysam, then preprocesses the data by imputing, embedding, and transforming the input data."""
        if self.args.vcf is not None:
            self.data_structure = DataStructure(self.args.vcf, dtype=self.dtype)
        self.data_structure.load_and_preprocess_data(self.args)

    def save_model(self, model, filename):
        """Saves the trained model to a file.

        Args:
            model (torch.nn.Module): The trained PyTorch model to save.
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(model.state_dict(), filename)
        if self.args.verbose >= 1:
            self.logger.info(f"Model saved to {filename}")

    @timer
    def train_rf(self, clf_params, objective_mode=False):
        """
        Trains an XGBRegressor model using the specified parameters and data loaders.

        The method supports data augmentation using SMOTE (Synthetic Minority Over-sampling Technique) and evaluates the model's performance using Root Mean Squared Error (RMSE).

        Args:
            clf_params (dict): Parameters for Random Forest or Gradient Boosting models.
            objective_mode (bool, optional): If True, the method is used for optimization objectives. Defaults to False.

        Returns:
            RandomForestRegressor or XGBRegressor: The trained model.
            float: RMSE of the model on the validation set.
            (additional returns if not in objective_mode): Additional data related to model training and evaluation.

        Notes:
            - The function first checks if SMOTE is to be applied and performs data augmentation accordingly.
            - Depending on the configuration, either a Random Forest or a Gradient Boosting model is trained.
            - The performance of the trained model is evaluated using RMSE on the validation dataset.
        """
        if self.args.verbose >= 2:
            self.logger.info("\n\n")

        X_train = self.data_structure.data["X_train"]
        y_train = self.data_structure.data["y_train"]
        sample_weights = self.data_structure.samples_weight

        centroids = None
        if self.args.oversample_method.lower() != "none":
            (
                features,
                labels,
                sample_weights,
                centroids,
                df,
                bins,
                centroids_orig,
                bins_resampled,
            ) = synthetic_resampling(
                X_train,
                y_train,
                sample_weights,
                self.args.n_bins,
                self.args,
                method=self.args.oversample_method,
                smote_neighbors=self.args.oversample_neighbors,
            )

            if features is None or labels is None:
                msg = "Synthetic data augmentation failed during optimization. Pruning trial."
                self.logger.warning(msg)
                if objective_mode:
                    return None
                else:
                    msg = "Synthetic data augmentation failed. Try adjusting the parameters supplied to SMOTE or SMOTER."
                    self.logger.error(msg)
                    raise ValueError(msg)

            if not objective_mode:
                self.visualize_oversampling(
                    features,
                    labels,
                    sample_weights,
                    df,
                    bins_resampled,
                )
        else:
            features = X_train.copy()
            labels = y_train.copy()

            if isinstance(sample_weights, torch.Tensor):
                sample_weights = sample_weights.numpy()
            sample_weights = sample_weights.copy()

        if self.args.use_gradient_boosting:
            X_train_val = self.data_structure.data["X_train_val"]
            y_train_val = self.data_structure.data["y_train_val"]
            X_val = self.data_structure.data["X_val"]
            y_val = self.data_structure.data["y_val"]

        sample_weights_val = np.ones((X_val.shape[0],))

        features, labels, sample_weights = validate_is_numpy(
            features, labels, sample_weights
        )
        X_val, y_val, sample_weights_val = validate_is_numpy(
            X_val, y_val, sample_weights_val
        )

        if self.args.use_gradient_boosting:
            callbacks = None
            clf_params["use_lr_scheduler"] = self.args.gb_use_lr_scheduler
            if clf_params["use_lr_scheduler"]:
                learning_rates = np.linspace(
                    clf_params["learning_rate"],
                    0.01,
                    self.args.gb_n_estimators,
                    endpoint=True,
                )
                lrs = xgb.callback.LearningRateScheduler(learning_rates.tolist())
                callbacks = [lrs]

        if "n_estimators" in clf_params:
            n_estimators = clf_params.pop("n_estimators")
        else:
            n_estimators = self.args.gb_n_estimators

        clf = xgb.XGBRegressor(
            n_estimators=n_estimators,
            multi_strategy=self.args.gb_multi_strategy,
            **clf_params,
            callbacks=callbacks,
            verbosity=0,
        )

        if self.args.use_gradient_boosting:
            clf.fit(
                features,
                labels,
                eval_metric=self.args.gb_eval_metric,
                eval_set=[(X_train, y_train), (X_train_val, y_train_val)],
                sample_weight=sample_weights,
                verbose=0,
            )
        else:
            clf.fit(features, labels, sample_weight=sample_weights)
        y_pred = clf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        if not objective_mode:
            self.data_structure.train_loader.dataset.features = torch.tensor(
                features, dtype=self.dtype
            )
            self.data_structure.train_loader.dataset.labels = torch.tensor(
                labels, dtype=self.dtype
            )

            self.data_structure.train_loader.dataset.sample_weights = torch.tensor(
                sample_weights, dtype=self.dtype
            )

        if objective_mode:
            return clf, rmse
        else:
            return clf, None, rmse, None, None, None, centroids

    def visualize_oversampling(
        self, features, labels, sample_weights, df, bins_resampled
    ):
        """
        Visualizes the effect of SMOTE (Synthetic Minority Over-sampling Technique) on the dataset.

        This method creates a visual comparison of the original and the oversampled datasets.

        Args:
            features (np.array or pandas.DataFrame): The feature set, either as a NumPy array or a DataFrame.
            labels (np.array or pandas.DataFrame): The label set, expected to contain 'x' and 'y' coordinates.
            sample_weights (np.array): Array of sample weights.
            df (pandas.DataFrame): Original DataFrame before applying SMOTE.
            bins_resampled (array-like): Array of bin labels for the data after applying SMOTE.

        Notes:
            - The method first validates and converts the features, labels, and sample weights to pandas DataFrames.
            - It then combines these into a single DataFrame and passes this to the `plot_smote_bins` method of the `PlotGenIE` class.
            - This visualization helps in understanding how SMOTE affects the distribution of samples across different geographical bins.
        """
        features, labels, sample_weights = validate_is_numpy(
            features, labels, sample_weights
        )

        labels = self.data_structure.norm.inverse_transform(labels)
        geo_coords_is_valid(labels)

        dfX = pd.DataFrame(features)
        dfy = pd.DataFrame(labels, columns=["x", "y"])
        dfX["sample_weights"] = sample_weights
        df_smote = pd.concat([dfX, dfy], axis=1)
        ytmp = df[["x", "y"]].to_numpy()
        ytmp = self.data_structure.norm.inverse_transform(ytmp)
        df["x"] = ytmp[:, 0]
        df["y"] = ytmp[:, 1]

        self.plotting.plot_smote_bins(
            df_smote,
            bins_resampled,
            df,
            self.args.n_bins,
            self.args.shapefile_url,
            buffer=self.args.bbox_buffer,
            marker_scale_factor=self.args.sample_point_scale,
        )

    @timer
    def train_model(
        self,
        train_loader,
        val_loader,
        model,
        optimizer,
        trial=None,
        objective_mode=False,
    ):
        """
        Train the PyTorch model with given parameters.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): PyTorch model to be trained.
            optimizer: Optimizer.
            trial (optuna.Trial): Current Optuna trial. Defaults to None.
            objective_mode (bool): Whether to return just the model for Optuna's objective function. Defaults to False.

        Returns:
            Model and training statistics, or just the model if in objective mode.
        """
        early_stopping = EarlyStopping(
            output_dir=self.args.output_dir,
            prefix=self.args.prefix,
            patience=self.args.early_stop_patience,
            verbose=self.args.verbose >= 2,
            delta=0,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
            verbose=self.args.verbose >= 2,
        )

        train_losses, val_losses, epoch_times = [], [], []
        rolling_window_size = 20
        total_time = 0

        epochs = self.args.max_epochs

        if self.args.verbose >= 2:
            self.logger.info("\n\n")

        centroids = None
        if self.args.oversample_method.lower() != "none":
            (
                features,
                labels,
                sample_weights,
                centroids,
                df,
                bins,
                centroids_orig,
                bins_resampled,
            ) = synthetic_resampling(
                train_loader.dataset.features,
                train_loader.dataset.labels,
                train_loader.dataset.sample_weights,
                self.args.n_bins,
                self.args,
                method=self.args.oversample_method,
                smote_neighbors=self.args.oversample_neighbors,
            )

            if features is None or labels is None:
                if objective_mode:
                    return None
                else:
                    msg = "Synthetic data augmentation failed. Try adjusting the parameters supplied to SMOTE or SMOTER."
                    self.logger.error(msg)
                    raise ValueError(msg)

            if not objective_mode:
                self.visualize_oversampling(
                    features,
                    labels,
                    sample_weights,
                    df,
                    bins_resampled,
                )

            train_dataset = CustomDataset(
                features, labels, sample_weights=sample_weights, dtype=self.dtype
            )

            kwargs = {"batch_size": train_loader.batch_size}

            if self.args.use_weighted == "none":
                sample_weights = torch.ones_like(
                    sample_weights, dtype=self.dtype, device=self.device
                )
                kwargs["shuffle"] = True
            else:
                if self.args.use_weighted in ["sampler", "both"]:
                    kwargs = self._reset_weighted_sampler(
                        sample_weights, train_dataset, kwargs
                    )

                if self.args.use_weighted in ["loss", "both"]:
                    if not isinstance(sample_weights, torch.Tensor):
                        sample_weights = torch.tensor(
                            sample_weights, dtype=self.dtype, device=self.device
                        )

                    if self.args.use_weighted != "both":
                        kwargs["shuffle"] = True

            train_loader = DataLoader(train_dataset, **kwargs)
            train_loader.dataset.sample_weights = sample_weights

        for epoch in range(epochs):
            # Training
            start_time, avg_train_loss = self.train_step(
                train_loader, model, optimizer, self.args.grad_clip, objective_mode
            )
            if avg_train_loss is None:
                # If errored out, then start_time will be string containing
                # exception.
                self.logger.warning(
                    f"Model training failed at epoch {epoch}: {start_time}"
                )
                return None, None
            train_losses.append(avg_train_loss)
            # Validation
            avg_val_loss = self.test_step(val_loader, model)
            val_losses.append(avg_val_loss)

            if self.args.verbose >= 2:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Final Val Loss: {avg_val_loss:.4f}"
                )

            end_time = time.time()
            epoch_duration = end_time - start_time
            epoch_times.append(epoch_duration)

            # Logging
            if self.args.verbose >= 2 and epoch % (rolling_window_size * 2) == 0:
                self.logger.info(
                    f"Current model train time ({epoch}/{epochs}): "
                    f"{total_time:.2f}s, - "
                    f"Rolling Average Time: "
                    f"{total_time/len(epoch_times):.2f} seconds"
                )

            # Early Stopping and LR Scheduler
            early_stopping(avg_val_loss, model)
            lr_scheduler.step(avg_val_loss)

            if early_stopping.early_stop:
                if self.args.verbose >= 2:
                    self.logger.info("Early stopping triggered.")
                break

            if trial is not None and trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        log_message = (
            f"Training Completed - "
            f"Final Train Loss: {train_losses[-1]:.4f} - "
            f"Final Val Loss: {val_losses[-1]:.4f}"
        )

        if self.args.verbose >= 2:
            self.logger.info(log_message)

        rolling_avgs, rolling_stds = self.compute_rolling_statistics(
            epoch_times, rolling_window_size
        )

        if objective_mode:
            return model, np.mean(val_losses)
        else:
            if self.args.oversample_method.lower() != "none":
                self.data_structure.train_loader.dataset.features = torch.tensor(
                    features, dtype=self.dtype
                )
                self.data_structure.train_loader.dataset.labels = torch.tensor(
                    labels, dtype=self.dtype
                )

                if sample_weights is not None:
                    self.data_structure.train_loader.dataset.sample_weights = (
                        torch.tensor(sample_weights, dtype=self.dtype)
                    )
                else:
                    self.data_structure.train_loader.dataset.sample_weights = (
                        torch.ones(sample_weights, dtype=self.dtype)
                    )
            return (
                model,
                train_losses,
                val_losses,
                rolling_avgs,
                rolling_stds,
                total_time,
                centroids,
            )

    def _reset_weighted_sampler(self, sample_weights, train_dataset, kwargs):
        weighted_sampler = self.data_structure.weighted_sampler
        weighted_sampler.indices = np.arange(len(train_dataset))
        weighted_sampler.weights = sample_weights
        kwargs["sampler"] = weighted_sampler
        return kwargs

    def train_step(self, train_loader, model, optimizer, grad_clip, objective_mode):
        """
        Executes a single training step (epoch) for the given model using the provided data loader, loss function, and optimizer.

        Args:
            train_loader (DataLoader): DataLoader providing the batched training data.
            model (torch.nn.Module): The neural network model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer used for model parameter updates.
            grad_clip (bool): Flag indicating whether gradient clipping should be applied.
            objective_mode (bool): Whether using objective mode.

        Returns:
            tuple: A tuple containing:
                - start_time (float): The start time of the training epoch.
                - avg_train_loss (float): The average training loss for the epoch.

        Notes:
            - The method iterates over batches from the train_loader, performing forward and backward passes, and updates the model parameters.
            - If grad_clip is True, it applies gradient clipping to prevent exploding gradients.
            - The method returns the start time of the training epoch and the average training loss.
        """
        start_time = time.time()  # Start time for the epoch
        model.train()
        total_loss = 0.0

        try:
            for batch in train_loader:
                data, targets, sample_weight = batch

                data = torch.tensor(data, dtype=self.dtype).to(model.device)
                targets = torch.tensor(targets, dtype=self.dtype).to(model.device)
                sample_weight = torch.tensor(sample_weight, dtype=self.dtype).to(
                    model.device
                )

                optimizer.zero_grad()
                outputs = model(data)
                loss = self.criterion(outputs, targets, sample_weight=sample_weight)
                loss.backward()

                if grad_clip:
                    # Gradient clipping.
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
        except Exception as e:
            if objective_mode:
                return str(e), None
            else:
                raise e
        return start_time, avg_train_loss

    def test_step(self, val_loader, model):
        """
        Executes a validation/test step for the given model using the provided data loader and loss function.

        Args:
            val_loader (DataLoader): DataLoader providing the batched validation or test data.
            model (torch.nn.Module): The neural network model to be evaluated.

        Returns:
            float: The average validation or test loss for the entire dataset.

        Notes:
            - The method iterates over batches from the val_loader, performing forward passes, and computes the loss.
            - It calculates the average loss over all batches, which is returned as the evaluation metric.
            - No gradient calculations or backpropagation are performed, as the model is set to evaluation mode.
        """
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, targets, sample_weight = batch
                data = torch.tensor(data, dtype=self.dtype).to(model.device)
                targets = torch.tensor(targets, dtype=self.dtype).to(model.device)
                sample_weight = torch.tensor(sample_weight, dtype=self.dtype).to(
                    model.device
                )

                outputs = model(data)
                val_loss = self.criterion(outputs, targets)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        return avg_val_loss

    def compute_rolling_statistics(self, times, window_size):
        """
        Computes rolling average and standard deviation over a specified window size.

        Args:
            times (list or array-like): A sequence of numerical values (e.g., times or scores).
            window_size (int): The number of elements to consider for each rolling window.

        Returns:
            tuple: A tuple containing two lists:
                - averages (list): The rolling averages.
                - std_devs (list): The rolling standard deviations.

        Notes:
            - This method is useful for analyzing time series data where you need to smooth out short-term fluctuations and highlight longer-term trends or cycles.
        """
        averages = []
        std_devs = []
        for i in range(len(times)):
            window = times[max(0, i - window_size + 1) : i + 1]
            avg = np.mean(window)
            std = np.std(window)
            averages.append(avg)
            std_devs.append(std)
        return averages, std_devs

    def predict_locations(
        self,
        model,
        data_loader,
        outfile,
        best_params,
        return_truths=False,
        use_rf=False,
    ):
        """
        Predict locations using the trained model and evaluate predictions.

        Args:
            args (argparse.Namespace): argparsed arguments from command-line.
            model (torch.nn.Module): Trained PyTorch model for predictions.
            data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for prediction.
            outfile (str): Output filename.
            best_params(dict): Dictionary with best parameters from Optuna search and/ or user-specified parameters.
            return_truths (bool): Whether to return truths as well as predictions. Defaults to False.
            use_rf (bool): Whether to use RandomForest or GradientBoosting models instead of deep learning model. Defaults to False.

        Returns:
            pandas.DataFrame: DataFrame with predicted locations and corresponding sample IDs.
        """
        if use_rf:
            predictions = model.predict(data_loader.dataset.features.numpy())
            ground_truth = data_loader.dataset.labels.numpy()
        else:
            predictions, ground_truth = self.model_predict(model, data_loader)
        sample_weights = data_loader.dataset.sample_weights.numpy()

        # Rescale predictions and ground truth to original scale
        predictions, ground_truth, metrics = self.calculate_prediction_metrics(
            outfile, predictions, ground_truth, sample_weights, best_params
        )

        if return_truths:
            return predictions, metrics, ground_truth
        return predictions, metrics

    def calculate_prediction_metrics(
        self, outfile, predictions, ground_truth, sample_weights, best_params
    ):
        def rescale_predictions(y):
            return self.data_structure.norm.inverse_transform(y)

        def mad(data):
            return np.median(np.abs(data - np.median(data)))

        def coefficient_of_variation(data):
            return np.std(data) / np.mean(data)

        def within_threshold(data, threshold):
            return np.mean(data < threshold)

        def rmse(predictions, targets):
            return np.sqrt(((predictions - targets) ** 2).mean())

        predictions = rescale_predictions(predictions)
        ground_truth = rescale_predictions(ground_truth)

        # Calculate Haversine error for each pair of points
        haversine_errors = haversine_distances_agg(ground_truth, predictions, np.array)

        geo_coords_is_valid(predictions)
        geo_coords_is_valid(ground_truth)

        # Evaluate predictions
        (
            rmse,
            mean_dist,
            median_dist,
            std_dist,
            spearman_corr_haversine,
            spearman_corr_x,
            spearman_corr_y,
            spearman_p_value_haversine,
            spearman_p_value_x,
            spearman_p_value_y,
            pearson_corr_haversine,
            pearson_corr_x,
            pearson_corr_y,
            pearson_p_value_haversine,
            pearson_p_value_x,
            pearson_p_value_y,
            rho,
            rho_p,
            haversine_mad,
            cv,
            iqr,
            percentiles,
            percentage_within_20km,
            percentage_within_50km,
            percentage_within_75km,
            z_scores,
            mean_absolute_z_score,
            haversine_rmse,
            ks,
            pval,
            skew,
        ) = self.get_all_stats(
            predictions, ground_truth, mad, coefficient_of_variation, within_threshold
        )

        self.print_stats_to_logger(
            mean_dist,
            median_dist,
            std_dist,
            spearman_corr_x,
            spearman_corr_y,
            spearman_p_value_x,
            spearman_p_value_y,
            pearson_corr_x,
            pearson_corr_y,
            pearson_p_value_x,
            pearson_p_value_y,
            rho,
            rho_p,
            haversine_mad,
            cv,
            iqr,
            percentiles,
            percentage_within_20km,
            percentage_within_50km,
            percentage_within_75km,
            mean_absolute_z_score,
            haversine_rmse,
            ks,
            pval,
            skew,
        )

        self.plotting.plot_error_distribution(haversine_errors, outfile)

        outfile2 = outfile.split("/")[-1]
        outfile2 = outfile2.split("_")
        for part in outfile2:
            if part.startswith("val"):
                dataset = "validation"
            elif part == "test":
                dataset = "test"

        outfile_cumulative_dist = (
            f"{self.args.prefix}_{dataset}_cumulative_error_distribution.png"
        )

        outfile_zscores = f"{self.args.prefix}_{dataset}_zscores.png"

        self.plotting.plot_cumulative_error_distribution(
            haversine_errors,
            outfile_cumulative_dist,
            percentiles,
            median_dist,
            mean_dist,
        )
        self.plotting.plot_zscores(z_scores, haversine_errors, outfile_zscores)

        # return the evaluation metrics along with the predictions
        metrics = {
            "root_mean_squared_error": rmse,
            "haversine_rmse": haversine_rmse,
            "mean_dist": mean_dist,
            "median_dist": median_dist,
            "stdev_dist": std_dist,
            "kolmogorov_smirnov": ks,
            "kolmogorov-smirnov_pval": pval,
            "skewness": skew,
            "rho": rho,  # Spearman's for each sample as a whole.
            "rho_p": rho_p,  # Spearman's for each sample as a whole.
            "spearman_corr_longitude": spearman_corr_x,
            "spearman_corr_latitude": spearman_corr_y,
            "spearman_pvalue_longitude": spearman_p_value_x,
            "spearman_pvalue_latitude": spearman_p_value_y,
            "pearson_corr_longitude": pearson_corr_x,
            "pearson_corr_latitude": pearson_corr_y,
            "pearson_pvalue_longitude": pearson_p_value_x,
            "pearson_pvalue_latitude": pearson_p_value_y,
            "mad_haversine": haversine_mad,
            "coeffecient_of_variation": cv,
            "interquartile_range": iqr,
            "percentile_25": percentiles[0],
            "percentile_50": percentiles[1],
            "percentiles_75": percentiles[2],
            "percent_within_20km": percentage_within_20km,
            "percent_within_50km": percentage_within_50km,
            "percent_within_75km": percentage_within_75km,
            "mean_absolute_z_score": mean_absolute_z_score,
        }

        return predictions, ground_truth, metrics

    def get_all_stats(
        self, predictions, ground_truth, mad, coefficient_of_variation, within_threshold
    ):
        rmse = calculate_rmse(predictions, ground_truth)
        mean_dist = haversine_distances_agg(ground_truth, predictions, np.mean)
        median_dist = haversine_distances_agg(ground_truth, predictions, np.median)
        std_dist = haversine_distances_agg(ground_truth, predictions, np.std)

        haversine_errors = haversine_distances_agg(ground_truth, predictions, np.array)

        # Ideal distances array - all zeros indicating no error
        ideal_distances = np.zeros_like(haversine_errors)

        (
            spearman_corr_haversine,
            spearman_corr_x,
            spearman_corr_y,
            spearman_p_value_haversine,
            spearman_p_value_x,
            spearman_p_value_y,
        ) = self.get_correlation_coef(
            predictions, ground_truth, haversine_errors, ideal_distances, spearmanr
        )
        (
            pearson_corr_haversine,
            pearson_corr_x,
            pearson_corr_y,
            pearson_p_value_haversine,
            pearson_p_value_x,
            pearson_p_value_y,
        ) = self.get_correlation_coef(
            predictions, ground_truth, haversine_errors, ideal_distances, pearsonr
        )

        rho, rho_p = spearmanr(predictions.ravel(), ground_truth.ravel())

        # Calculate median absolute deviation for Haversine distances
        haversine_mad = mad(haversine_errors)

        cv = coefficient_of_variation(haversine_errors)

        # Inter-quartile range.
        iqr = np.percentile(haversine_errors, 75) - np.percentile(haversine_errors, 25)

        percentiles = np.percentile(haversine_errors, [25, 50, 75])

        # Percentage of predictions within <N> km error
        percentage_within_20km = within_threshold(haversine_errors, 25) * 100
        percentage_within_50km = within_threshold(haversine_errors, 50) * 100
        percentage_within_75km = within_threshold(haversine_errors, 75) * 100

        z_scores = (haversine_errors - np.mean(haversine_errors)) / np.std(
            haversine_errors
        )

        mean_absolute_z_score = np.mean(np.abs(z_scores))

        haversine_rmse = mean_squared_error(
            haversine_errors.reshape(-1, 1),
            np.zeros_like(haversine_errors).reshape(-1, 1),
            squared=False,
        )

        # 0 is best, negative means overestimations, positive means
        # underestimations
        ks, pval, skew = kstest(ground_truth, predictions)
        return (
            rmse,
            mean_dist,
            median_dist,
            std_dist,
            spearman_corr_haversine,
            spearman_corr_x,
            spearman_corr_y,
            spearman_p_value_haversine,
            spearman_p_value_x,
            spearman_p_value_y,
            pearson_corr_haversine,
            pearson_corr_x,
            pearson_corr_y,
            pearson_p_value_haversine,
            pearson_p_value_x,
            pearson_p_value_y,
            rho,
            rho_p,
            haversine_mad,
            cv,
            iqr,
            percentiles,
            percentage_within_20km,
            percentage_within_50km,
            percentage_within_75km,
            z_scores,
            mean_absolute_z_score,
            haversine_rmse,
            ks,
            pval,
            skew,
        )

    def print_stats_to_logger(
        self,
        mean_dist,
        median_dist,
        std_dist,
        spearman_corr_x,
        spearman_corr_y,
        spearman_p_value_x,
        spearman_p_value_y,
        pearson_corr_x,
        pearson_corr_y,
        pearson_p_value_x,
        pearson_p_value_y,
        rho,
        rho_p,
        haversine_mad,
        cv,
        iqr,
        percentiles,
        percentage_within_20km,
        percentage_within_50km,
        percentage_within_75km,
        mean_absolute_z_score,
        haversine_rmse,
        ks,
        pval,
        skew,
    ):
        self.logger.info(f"Validation Haversine Error (km) = {mean_dist}")
        self.logger.info(f"Median Validation Error (km) = {median_dist}")
        self.logger.info(f"Standard deviation for Haversine Error (km) = {std_dist}")

        self.logger.info(f"Root Mean Squared Error (km) = {haversine_rmse}")
        self.logger.info(
            f"Median Absolute Deviation of Prediction Error (km) = {haversine_mad}"
        )
        self.logger.info(f"Coeffiecient of Variation for Prediction Error = {cv}")
        self.logger.info(f"Interquartile Range of Prediction Error (km) = {iqr}")

        for perc, output in zip([25, 50, 75], percentiles):
            self.logger.info(f"{perc} percentile of prediction error (km) = {output}")

        self.logger.info(
            f"Percentage of samples with error within 20 km = {percentage_within_20km}"
        )
        self.logger.info(
            f"Percentage of samples with error within 50 km = {percentage_within_50km}"
        )
        self.logger.info(
            f"Percentage of samples with error within 75 km = {percentage_within_75km}"
        )

        self.logger.info(
            f"Mean Absolute Z-scores of Prediction Error (km) = {mean_absolute_z_score}"
        )

        self.logger.info(
            f"Spearman's Correlation Coefficient = {rho}, P-value = {rho_p}"
        )
        self.logger.info(
            f"Spearman's Correlation Coefficient for Longitude = {spearman_corr_x}, P-value = {spearman_p_value_x}"
        )
        self.logger.info(
            f"Spearman's Correlation Coefficient for Latitude = {spearman_corr_y}, P-value = {spearman_p_value_y}"
        )
        self.logger.info(
            f"Pearson's Correlation Coefficient for Longitude = {pearson_corr_x}, P-value = {pearson_p_value_x}"
        )
        self.logger.info(
            f"Pearson's Correlation Coefficient for Latitude = {pearson_corr_y}, P-value = {pearson_p_value_y}"
        )

        # 0 is best, positive means more undeerestimations
        # negative means more overestimations.
        self.logger.info(f"Skewness = {skew}")

        # Goodness of fit test.
        # Small P-value means poor fit.
        # I.e., significantly deviates from reference distribution.
        self.logger.info(f"Kolmogorov-Smirnov Test = {ks}, P-value = {pval}")

    def get_correlation_coef(
        self, predictions, ground_truth, haversine_errors, ideal_distances, corr_func
    ):
        corr_haversine, p_value_haversine = spearmanr(haversine_errors, ideal_distances)

        corr_x, p_value_x = corr_func(predictions[:, 0], ground_truth[:, 0])

        corr_y, p_value_y = corr_func(predictions[:, 1], ground_truth[:, 1])
        return corr_haversine, corr_x, corr_y, p_value_haversine, p_value_x, p_value_y

    def model_predict(self, model, data_loader):
        model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for data, target, _ in data_loader:
                data = torch.tensor(data, dtype=self.dtype).to(self.device)
                output = model(data)
                predictions.append(output.cpu().numpy())
                ground_truth.append(target.numpy())
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        return predictions, ground_truth

    def plot_bootstrap_aggregates(self, df, train_times):
        self.plotting.plot_bootstrap_aggregates(df)

        avg_time, stddev_time = self.compute_rolling_statistics(
            train_times, window_size=10
        )

        self.plotting.plot_times(
            avg_time,
            stddev_time,
            os.path.join(
                self.args.output_dir,
                "plots",
                f"{self.args.prefix}_avg_train_time.png",
            ),
        )

    def perform_standard_training(
        self, train_loader, val_loader, device, best_params, ModelClass
    ):
        """
        Perform standard model training.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            device (torch.device): Device for training ('cpu' or 'cuda').
            best_params (dict): Dictionary of parameters to use with model training.
            ModelClass (torch.nn.Module): Callable subclass for PyTorch model.
            criterion (callable): PyTorch callable loss function.

        Returns:
            A tuple containing the best model, training losses, and validation losses.
        """
        try:
            if self.args.verbose >= 1:
                self.logger.info("Starting standard model training.")

            if ModelClass != "GB":
                # Initialize the model
                model = ModelClass(
                    input_size=train_loader.dataset.features.shape[1],
                    width=best_params["width"],
                    nlayers=best_params["nlayers"],
                    dropout_prop=best_params["dropout_prop"],
                    device=device,
                    output_width=train_loader.dataset.labels.shape[1],
                    dtype=self.dtype,
                ).to(device)

                optimizer = self.extract_best_params(best_params, model)
                # train_loader = self.get_sample_weights(train_loader, best_params)

                # Train the model
                (
                    trained_model,
                    train_losses,
                    val_losses,
                    _,
                    __,
                    total_train_time,
                    centroids,
                ) = self.train_model(
                    train_loader,
                    val_loader,
                    model,
                    optimizer,
                    trial=None,
                    objective_mode=False,
                )

            else:
                start_time = time.time()
                # train_loader = self.get_sample_weights(train_loader, best_params)
                (
                    trained_model,
                    _,
                    val_losses,
                    __,
                    ___,
                    ____,
                    centroids,
                ) = self.train_rf(best_params, objective_mode=False)

                end_time = time.time()
                total_train_time = end_time - start_time

            if self.args.verbose >= 1:
                self.logger.info(
                    f"Standard training completed in {total_train_time / 60} "
                    f"minutes",
                )

            if ModelClass != "GB":
                return trained_model, train_losses, val_losses, centroids
            return trained_model, None, val_losses, centroids

        except Exception as e:
            self.logger.error(
                f"Unexpected error in perform_standard_training: {e}",
            )
            raise e

    def extract_best_params(self, best_params, model):
        lr = best_params["lr"] if "lr" in best_params else best_params["learning_rate"]

        l2 = (
            best_params["l2_weight"]
            if "l2_weight" in best_params
            else best_params["l2_reg"]
        )

        # Define the criterion and optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
        return optimizer

    def write_pred_locations(
        self,
        pred_locations,
        pred_indices,
        sample_data,
        filename,
    ):
        """write predicted locations to file."""
        if self.args.verbose >= 1:
            self.logger.info("Writing predicted coorinates to dataframe.")

        pred_locations_df = pd.DataFrame(pred_locations, columns=["x", "y"])
        sample_data = sample_data.reset_index()
        sample_data = sample_data.iloc[pred_indices].copy()

        pred_locations_df.reset_index(drop=True)
        pred_locations_df["sampleID"] = sample_data["sampleID"].tolist()
        pred_locations_df = pred_locations_df[["sampleID", "x", "y"]]
        pred_locations_df.to_csv(filename, header=True, index=False)
        return pred_locations_df

    def load_best_params(self, filename):
        if not Path(filename).is_file():
            msg = f"Could not find file storing best params: {filename}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        with open(filename, "r") as fin:
            best_params = json.load(fin)

        if not isinstance(best_params, dict):
            msg = f"Invalid format detected for best parameters object. Expected dict, but got: {type(best_params)}"
            self.logger.error(msg)
            raise TypeError(msg)
        return best_params

    def optimize_parameters(self, ModelClass):
        """
        Perform parameter optimization using Optuna.

        Args:
            criterion: The loss function to be used for the model.
            ModelClass (torch.nn.Module): The PyTorch model class for which the optimization is to be done.

        Returns:
            dict: Best parameters found by Optuna optimization.
        """
        if not self.args.do_gridsearch:
            self.logger.info("Optuna parameter search is not enabled.")
            return None

        if self.args.verbose >= 1:
            self.logger.info("Starting Optuna parameter search.")

        opt = Optimize(
            self.data_structure.train_loader,
            self.data_structure.val_loader,
            self.data_structure.test_loader,
            self.data_structure.samples_weight,
            self.data_structure.densities,
            self.data_structure.weighted_sampler,
            self.device,
            self.args,
            show_progress_bar=False,
            n_startup_trials=10,
            dtype=self.dtype,
        )

        if self.args.use_random_forest or self.args.use_gradient_boosting:
            best_trial, study = opt.perform_optuna_optimization(
                self.criterion, ModelClass, self.train_rf
            )
        else:
            best_trial, study = opt.perform_optuna_optimization(
                self.criterion, ModelClass, self.train_model
            )
        opt.process_optuna_results(study, best_trial)

        if self.args.verbose >= 1:
            self.logger.info("Optuna optimization completed!")

        return best_trial.params

    def perform_bootstrap_training(self, ModelClass, best_params):
        """
        Perform bootstrap training using the provided parameters.

        Args:
            criterion: The loss function to be used for the model.
            ModelClass (torch.nn.Module): The PyTorch model class to use.
            best_params (dict): Dictionary of best parameters found by Optuna or specified by the user.
        """
        if not self.args.bootstrap:
            self.logger.warning("Bootstrap training is not enabled.")
            return

        if self.args.verbose >= 1:
            self.logger.info("Starting bootstrap training.")

        boot = Bootstrap(
            self.data_structure.train_loader,
            self.data_structure.val_loader,
            self.data_structure.indices["val_indices"],
            self.args.nboots,
            self.args.max_epochs,
            self.device,
            self.data_structure.samples_weight,
            best_params["width"],
            best_params["nlayers"],
            best_params["dropout_prop"],
            best_params["learning_rate"],
            best_params["l2_reg"],
            self.args.early_stop_patience,
            self.args.output_dir,
            self.args.prefix,
            self.data_structure.sample_data,
            verbose=self.args.verbose,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
        )

        boot.perform_bootstrap_training(
            self.train_model,
            self.predict_locations,
            self.write_pred_locations,
            ModelClass,
            self.criterion,
            self.data_structure.coord_scaler,
        )

        if self.args.verbose >= 1:
            self.logger.info("Bootstrap training completed.")

    def evaluate_and_save_results(
        self,
        model,
        train_losses,
        val_losses,
        best_params,
        dataset="val",
        centroids=None,
        use_rf=False,
    ):
        """
        Evaluate the model and save the results.

        Args:
            model: The trained model to evaluate.
            train_losses: List of training losses.
            val_losses: List of validation losses.
            best_params (dict): Dictionary of best parameters from Optuna search and/ or user-defined parameters.
            dataset (str): Whether 'val' or 'test' dataset.
            centroids (np.ndarray): Centroids if using synthetic resampling with 'kmeans' or 'optics' options.; otherwise None. Defaults to None.
            use_rf (bool): Whether to use RandomForest model. If False, uses deep learning model instead. Defaults to False (deep learning model).
        """
        if self.args.verbose >= 1:
            self.logger.info(f"Evaluating the model on the {dataset} set.")

        if dataset not in ["val", "test"]:
            self.logger.error(
                "Only 'val' or 'test' are supported for the 'dataset' option."
            )
            raise ValueError(
                "Only 'val' or 'test' are supported for the 'dataset' option."
            )

        outdir = self.args.output_dir
        prefix = self.args.prefix

        middir = dataset
        if dataset.startswith("val"):
            middir = dataset + "idation" if dataset.endswith("val") else dataset
            loader = self.data_structure.val_loader
        else:
            loader = self.data_structure.test_loader

        y_train = self.data_structure.train_loader.dataset.labels.numpy()
        X_train = self.data_structure.train_loader.dataset.features.numpy()
        y_train = self.data_structure.norm.inverse_transform(y_train)
        if centroids is not None:
            centroids = self.data_structure.norm.inverse_transform(centroids)

        if use_rf:
            y_train_pred = model.predict(X_train)
        else:
            y_train_pred = model(torch.tensor(X_train, dtype=self.dtype))
            y_train_pred = y_train_pred.detach().numpy()

        val_errordist_outfile = os.path.join(
            outdir, "plots", f"{prefix}_{middir}_error_distributions.png"
        )

        val_preds, val_metrics, y_true = self.predict_locations(
            model,
            loader,
            val_errordist_outfile,
            best_params,
            return_truths=True,
            use_rf=use_rf,
        )

        geo_coords_is_valid(val_preds)
        geo_coords_is_valid(y_true)

        # Save validation results to file
        val_metric_outfile = os.path.join(
            outdir, middir, f"{prefix}_{middir}_metrics.json"
        )

        val_preds_outfile = os.path.join(
            outdir, middir, f"{prefix}_{middir}_predictions.txt"
        )

        val_preds_df = self.write_pred_locations(
            val_preds,
            self.data_structure.indices[f"{dataset}_indices"],
            self.data_structure.sample_data,
            val_preds_outfile,
        )

        with open(val_metric_outfile, "w") as fout:
            json.dump({k: v for k, v in val_metrics.items()}, fout, indent=2)

        if self.args.verbose >= 1:
            self.logger.info("Validation metrics saved.")

        if dataset.startswith("val"):
            # Save training and validation losses to file
            train_outfile = os.path.join(
                outdir, "training", f"{prefix}_train_{dataset}_results.json"
            )
            training_results = {
                "train_losses": train_losses,
                f"{dataset}_losses": val_losses,
            }
        else:
            train_outfile = os.path.join(
                outdir, "training", f"{prefix}_train_results.json"
            )
            training_results = {"train_losses": train_losses}

        with open(train_outfile, "w") as fout:
            try:
                json.dump(training_results, fout, indent=2)
            except TypeError:
                pass

        if self.args.verbose >= 1:
            self.logger.info("Training and validation losses saved.")

        # Plot training history
        if not use_rf:
            self.plotting.plot_history(train_losses, val_losses)
        self.plotting.plot_geographic_error_distribution(
            y_true,
            val_preds,
            self.args.shapefile_url,
            dataset,
            buffer=self.args.bbox_buffer,
            marker_scale_factor=self.args.sample_point_scale,
            min_colorscale=self.args.min_colorscale,
            max_colorscale=self.args.max_colorscale,
            n_contour_levels=self.args.n_contour_levels,
            centroids=centroids,
        )

        self.plotting.plot_scatter_samples_map(
            y_train,
            y_true,
            dataset,
            self.args.shapefile_url,
            buffer=self.args.bbox_buffer,
        )

        self.plotting.polynomial_regression_plot(
            y_true, val_preds, dataset, dtype=self.dtype
        )

        if self.args.verbose >= 1:
            self.logger.info("Training history plotted.")

    def make_unseen_predictions(self, model, device, use_rf=False):
        if self.args.verbose >= 1:
            # Predictions on unseen data
            self.logger.info("Making predictions on unseen data...")

        outdir = self.args.output_dir
        prefix = self.args.prefix

        if not use_rf:
            dtype = self.dtype

            # Convert X_pred to a PyTorch tensor and move it to the correct
            # device (GPU or CPU)
            pred_tensor = torch.tensor(
                self.data_structure.data["X_pred"], dtype=dtype
            ).to(device)

            with torch.no_grad():
                # Make predictions
                pred_locations_scaled = model(pred_tensor)

            pred_locations = self.data_structure.norm.inverse_transform(
                pred_locations_scaled.cpu().numpy()
            )
        else:
            pred_locations_scaled = model.predict(self.data_structure.data["X_pred"])
            pred_locations = self.data_structure.norm.inverse_transform(
                pred_locations_scaled
            )

        pred_outfile = os.path.join(
            outdir,
            "predictions",
            f"{prefix}_predictions.txt",
        )

        real_preds = self.write_pred_locations(
            pred_locations,
            self.data_structure.pred_indices,
            self.data_structure.sample_data,
            pred_outfile,
        )

        return real_preds

    def train_test_predict(self):
        # Set seed and GPU
        if self.args.seed is not None:
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

        if self.args.gpu_number is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_number)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            device = torch.device("cpu")

        if self.args.verbose >= 1:
            self.logger.info(f"Using device: {device}")

        if self.args.verbose >= 2:
            self.logger.info("Creating output directory structure.")

        outdir = self.args.output_dir
        prefix = self.args.prefix

        try:
            # Creates DataStructure instance.
            # Loads and preprocesses data.
            self.load_data()
            self.data_structure.define_params(self.args)
            best_params = self.data_structure.params

            self.criterion = weighted_rmse_loss

            modelclass = "GB" if self.args.use_gradient_boosting else MLPRegressor

            # Parameter optimization with Optuna
            if self.args.do_gridsearch:
                if self.args.load_best_params is not None:
                    self.logger.warning(
                        "--load_best_params was specified; skipping paramter optimization and loading best parameeters."
                    )
                else:
                    best_params = self.optimize_parameters(ModelClass=modelclass)
                    self.data_structure.params = best_params
                    # Add only new keys from data_structure.params to
                    # best_params
                    for key, value in self.data_structure.params.items():
                        if key not in best_params:
                            best_params[key] = value
                    if self.args.verbose >= 1:
                        self.logger.info(f"Best found parameters: {best_params}")

            if self.args.load_best_params is not None:
                best_params = self.load_best_params(self.args.load_best_params)
                self.data_structure.params = best_params

                # Add only new keys from data_structure.params to best_params
                for key, value in self.data_structure.params.items():
                    if key not in best_params:
                        best_params[key] = value
                if self.args.verbose >= 1:
                    self.logger.info(
                        f"Best parameters loaded from parent directory {self.args.load_best_params}: {best_params}"
                    )

            # Model Training
            if self.args.do_bootstrap:
                self.perform_bootstrap_training(modelclass, best_params)

            (
                best_model,
                train_losses,
                val_losses,
                centroids,
            ) = self.perform_standard_training(
                self.data_structure.train_loader,
                self.data_structure.test_loader,
                device,
                best_params,
                modelclass,
            )

            use_rf = True if modelclass in ["RF", "GB"] else False
            self.evaluate_and_save_results(
                best_model,
                train_losses,
                val_losses,
                best_params,
                dataset="val",
                centroids=centroids,
                use_rf=use_rf,
            )

            self.evaluate_and_save_results(
                best_model,
                train_losses,
                val_losses,
                best_params,
                dataset="test",
                centroids=centroids,
                use_rf=use_rf,
            )

            real_preds = self.make_unseen_predictions(best_model, device, use_rf)

            model_out = os.path.join(
                outdir,
                "models",
                f"{prefix}_trained_model.pt",
            )

            if self.args.verbose >= 1:
                self.logger.info("Process completed successfully!.")
                self.logger.info(f"Saving model to: {model_out}")

            if not use_rf:
                self.save_model(best_model, model_out)

            save_execution_times(
                f"{self.args.output_dir}/benchmarking/{self.args.prefix}_execution_times.csv"
            )
            execution_times.clear()

        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()
            raise
