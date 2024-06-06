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
from sklearn.metrics import mean_squared_error

from geogenie.models.models import MLPRegressor
from geogenie.optimize.bootstrap import Bootstrap
from geogenie.optimize.optuna_opt import Optimize
from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.interpolate import run_genotype_interpolator
from geogenie.samplers.samplers import synthetic_resampling
from geogenie.utils.callbacks import callback_init
from geogenie.utils.data import UnlabeledDataset
from geogenie.utils.data_structure import DataStructure
from geogenie.utils.loss import WeightedDRMSLoss, WeightedHuberLoss, weighted_rmse_loss
from geogenie.utils.scorers import calculate_rmse, kstest
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
            "plots/shapefile",
            "benchmarking",
            "bootstrap_predictions",
            "bootstrap_metrics",
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
            self.args.basemap_fips,
            self.args.highlight_basemap_counties,
            self.args.shapefile,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
            filetype=self.args.filetype,
            dpi=self.args.plot_dpi,
            remove_splines=self.args.remove_splines,
        )

        self.boot = None

    def load_data(self):
        """Loads genotypes from VCF file using pysam, then preprocesses the data by imputing, embedding, and transforming the input data."""
        if self.args.vcf is not None:
            self.ds = DataStructure(self.args.vcf, dtype=self.dtype)
        self.ds.load_and_preprocess_data(self.args)

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

        X_train = self.ds.data["X_train"]
        y_train = self.ds.data["y_train"]
        sample_weights = self.ds.samples_weight

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
            X_train_val = self.ds.data["X_train_val"]
            y_train_val = self.ds.data["y_train_val"]
            X_val = self.ds.data["X_val"]
            y_val = self.ds.data["y_val"]

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
            self.ds.train_loader.dataset.features = torch.tensor(
                features, dtype=self.dtype
            )
            self.ds.train_loader.dataset.labels = torch.tensor(labels, dtype=self.dtype)

            self.ds.train_loader.dataset.sample_weights = torch.tensor(
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

        labels = self.ds.norm.inverse_transform(labels)
        geo_coords_is_valid(labels)

        dfX = pd.DataFrame(features)
        dfy = pd.DataFrame(labels, columns=["x", "y"])
        dfX["sample_weights"] = sample_weights
        df_smote = pd.concat([dfX, dfy], axis=1)
        ytmp = df[["x", "y"]].to_numpy()
        ytmp = self.ds.norm.inverse_transform(ytmp)
        df["x"] = ytmp[:, 0]
        df["y"] = ytmp[:, 1]

        self.plotting.plot_smote_bins(df_smote, bins_resampled, df, self.args.n_bins)

    @timer
    def train_model(
        self,
        train_loader,
        val_loader,
        model,
        optimizer,
        trial=None,
        objective_mode=False,
        do_bootstrap=False,
        early_stopping=None,
        lr_scheduler=None,
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
            do_bootstrap (bool): True if iin botstrap mode. False otherwise. Defaults to False.
            early_stopping (EarlyStopping): Early stopping callback to use with training. Must be provided if objective_mode is True. Defaults to None.
            lr_scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler to use with training. Must be provided if ``objective_mode=True``\. Defaults to None.

        Returns:
            Model and training statistics, or just the model if in objective mode.
        """
        if early_stopping is None and (objective_mode or do_bootstrap):
            msg = "Must provide 'early_stopping' argument if 'objective_mode is True', but got NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

        if lr_scheduler is None and (objective_mode or do_bootstrap):
            msg = "Must provide 'lr_scheduler' argument if 'objective_mode=True', but got NoneType."
            self.logger.error(msg)
            raise TypeError(msg)

        if not objective_mode:
            early_stopping, lr_scheduler = callback_init(optimizer, self.args)

        train_losses, val_losses = [], []

        self.train_loader_interp = train_loader
        centroids = None
        if (
            self.args.oversample_method.lower() != "none"
            and not do_bootstrap
            and not objective_mode
        ):
            (
                train_loader,
                centroids,
                features,
                labels,
                sample_weights,
            ) = run_genotype_interpolator(
                train_loader, self.args, self.ds, self.dtype, self.plotting
            )

            self.train_loader_interp = train_loader

        for epoch in range(self.args.max_epochs):
            # Training
            avg_train_loss = self.train_step(
                train_loader, model, optimizer, self.args.grad_clip, objective_mode
            )
            if avg_train_loss is None or np.isnan(avg_train_loss):
                # If errored out, then start_time will be string containing
                # exception.
                self.logger.warning(f"Model training failed at epoch {epoch}")

                if objective_mode and trial is not None:
                    raise optuna.exceptions.TrialPruned()

                if do_bootstrap and not objective_mode:
                    msg = (
                        f"Model training failed at epoch {epoch} during bootstrapping."
                    )
                    self.logger.error(msg)
                    return None
                return None, None
            train_losses.append(avg_train_loss)

            # Validation
            avg_val_loss = self.test_step(val_loader, model)
            val_losses.append(avg_val_loss)

            # Early Stopping and LR Scheduler
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                if self.args.verbose >= 2:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            lr_scheduler.step(avg_val_loss)

            if trial is not None and trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if objective_mode or do_bootstrap:
            return model

        if self.args.oversample_method.lower() != "none":
            features = torch.tensor(features, dtype=self.dtype)
            labels = torch.tensor(labels, dtype=self.dtype)
            if sample_weights is not None:
                sample_weights = torch.tensor(sample_weights, dtype=self.dtype)
            else:
                sample_weights = torch.ones(
                    (features.numpy().shape[0]), dtype=self.dtype
                )
            self.ds.train_loader.dataset.features = features
            self.ds.train_loader.dataset.labels = labels
            self.ds.train_loader.dataset.sample_weights = sample_weights

        return model, train_losses, val_losses, centroids

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
            float: The average training loss for the epoch.

        Notes:
            - The method iterates over batches from the train_loader, performing forward and backward passes, and updates the model parameters.
            - If grad_clip is True, it applies gradient clipping to prevent exploding gradients.
            - The method returns the average training loss.
        """
        model.train()
        total_loss = []

        total_loss = 0

        for batch in train_loader:
            data, targets, sample_weight = self._batch_init(model, batch)
            optimizer.zero_grad()

            try:
                outputs = model(data)
                loss = self.criterion(outputs, targets, sample_weight=sample_weight)
                loss.backward()
                if grad_clip:
                    # Gradient clipping.
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
            except Exception as e:
                if objective_mode:
                    self.logger.warning(f"Optuna Trial failed: {str(e)}")
                    return None
                else:
                    raise e
            total_loss = total_loss + loss.item()
        return total_loss / len(train_loader)

    def _batch_init(self, model, batch):
        data, targets, sample_weight = batch

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=self.dtype)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=self.dtype)
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight, dtype=self.dtype)
        return (
            data.to(model.device),
            targets.to(model.device),
            sample_weight.to(model.device),
        )

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
        total_val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                data, targets, _ = self._batch_init(model, batch)
                outputs = model(data)
                val_loss = self.criterion(outputs, targets)
                total_val_loss.append(val_loss.item())
        avg_val_loss = np.mean(total_val_loss)
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
        return_truths=False,
        use_rf=False,
        log_metrics=True,
        bootstrap=False,
        is_train=False,
        dataset=None,
        is_val=True,
    ):
        """Predict locations using the trained model and evaluate predictions.

        Args:
            args (argparse.Namespace): argparsed arguments from command-line.
            model (torch.nn.Module): Trained PyTorch model for predictions.
            data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for prediction.
            outfile (str): Output filename.
            return_truths (bool): Whether to return truths as well as predictions. Defaults to False.
            use_rf (bool): Whether to use RandomForest or GradientBoosting models instead of deep learning model. Defaults to False.
            log_metrics (bool): Whether to log metrics to STDOUT and STDERR.
            bootstrap (bool): Whether doing bootstrapping. If True, then will not make plots or print stats to log (but will still record stats in JSON files). Defaults to False.
            is_train (bool): Whether using train dataset. If True, does not make some of the plots for train dataset. Defaults to False.
            dataset (str): "test" or "val". Defaults to None.
            is_val (bool): Whether using validation/ test dataset. Otherwise using pred dataset. Defaults to True.

        Returns:
            pandas.DataFrame: DataFrame with predicted locations and corresponding sample IDs.
        """
        if use_rf:
            predictions = model.predict(data_loader.dataset.features.numpy())
            ground_truth = data_loader.dataset.labels.numpy()
        else:
            if is_val:
                predictions, ground_truth = self.model_predict(
                    model, data_loader, is_val=is_val
                )
            else:
                predictions = self.model_predict(model, data_loader, is_val=is_val)

        if is_val:
            # Rescale predictions and ground truth to original scale
            predictions, ground_truth, metrics = self.calculate_prediction_metrics(
                outfile,
                predictions,
                ground_truth,
                log_metrics,
                bootstrap,
                is_train,
                dataset,
            )

        if return_truths and is_val:
            return predictions, metrics, ground_truth
        elif not return_truths and is_val:
            return predictions, metrics
        else:
            return predictions

    def calculate_prediction_metrics(
        self,
        outfile,
        predictions,
        ground_truth,
        log_stats,
        bootstrap,
        is_train=False,
        dataset=None,
    ):
        def rescale_predictions(y):
            return self.ds.norm.inverse_transform(y)

        def mad(data):
            return np.median(np.abs(data - np.median(data)))

        def coefficient_of_variation(data):
            return np.std(data) / np.mean(data)

        def within_threshold(data, threshold):
            return np.mean(data < threshold)

        ground_truth = rescale_predictions(ground_truth)
        geo_coords_is_valid(ground_truth)

        predictions = rescale_predictions(predictions)
        geo_coords_is_valid(predictions)

        # Evaluate predictions
        metrics = self.get_all_stats(
            predictions,
            ground_truth,
            mad,
            coefficient_of_variation,
            within_threshold,
        )

        z_scores = metrics[0]
        values = metrics[1]
        haversine_errors = metrics[2]

        # return the evaluation metrics along with the predictions
        metrics_dict = self._create_metrics_dictionary(values)

        if log_stats and not bootstrap:
            self.print_stats_to_logger(metrics_dict)

        if self.boot is not None and not bootstrap:
            z_scores = (haversine_errors - np.mean(haversine_errors)) / np.std(
                haversine_errors
            )

        if not bootstrap and not is_train:
            self.plotting.plot_error_distribution(haversine_errors, outfile)

            outfile2 = outfile.split("/")[-1]
            outfile2 = outfile2.split("_")
            for part in outfile2:
                if part.startswith("val"):
                    dataset = "validation"
                elif part == "test":
                    dataset = "test"
                else:
                    dataset = "train"

            outfile_cumulative_dist = (
                f"{self.args.prefix}_{dataset}_cumulative_error_distribution.png"
            )

            outfile_zscores = f"{self.args.prefix}_{dataset}_zscores.png"

            self.plotting.plot_cumulative_error_distribution(
                haversine_errors,
                outfile_cumulative_dist,
                np.array(
                    [
                        metrics_dict["percentile_25"],
                        metrics_dict["percentile_50"],
                        metrics_dict["percentile_75"],
                    ]
                ),  # percentiles np.ndarray
                metrics_dict["median_dist"],
                metrics_dict["mean_dist"],
            )
            self.plotting.plot_zscores(z_scores, outfile_zscores)

        return predictions, ground_truth, metrics_dict

    def _aggregate_test_metrics(self, dataset):
        """
        Analyzes the given data to calculate mean, median, standard deviation, and 95% CI for each column.

        Args:
            dataset (str): Which dataset to use: {"test", "val"}.

        Returns:
            dict: Dictionary with column names as keys and their means as values.
        """
        pth = Path(self.args.output_dir)
        pth = pth.joinpath("bootstrap_metrics", dataset)
        pth.mkdir(exist_ok=True, parents=True)
        of = f"{self.args.prefix}_bootstrap_{dataset}_metrics.csv"
        infile = pth.joinpath(of)

        # Reading data into DataFrame
        df = pd.read_csv(infile)

        # Initializing dictionary for results
        results = {}

        # Initializing an empty DataFrame for aggregated data
        aggregated_data = pd.DataFrame()

        for column in df.columns:
            # Calculating mean, median, std
            mean = df[column].mean()
            median = df[column].median()
            std = df[column].std()

            # Calculating 95% confidence interval
            ci = 1.96 * (std / np.sqrt(len(df)))

            # Adding to results dictionary
            results[column] = mean

            # Adding to aggregated DataFrame
            aggregated_data.at["mean", column] = mean
            aggregated_data.at["median", column] = median
            aggregated_data.at["std", column] = std
            aggregated_data.at["95% CI lower", column] = mean - ci
            aggregated_data.at["95% CI upper", column] = mean + ci

        pth = Path(self.args.output_dir)
        pth = pth.joinpath("bootstrap_summaries")
        pth.mkdir(exist_ok=True, parents=True)
        of = f"aggregated_bootstrap_{dataset}_metrics.csv"
        outfile = pth.joinpath(of)

        # Writing aggregated data to CSV
        aggregated_data.to_csv(outfile, header=True, index=False)

        return results

    def _create_metrics_dictionary(self, values):
        """
        Creates a dictionary for metrics from a list of values.

        Args:
        values (list): List of values in the specified order, with 'percentiles' being a NumPy array at index 16.

        Returns:
        dict: Dictionary with metrics.
        """

        # List of keys for the dictionary
        keys = [
            "root_mean_squared_error",
            "mean_dist",
            "median_dist",
            "stdev_dist",
            "kolmogorov_smirnov",
            "kolmogorov_smirnov_pval",
            "skewness",
            "rho",
            "rho_p",
            "spearman_corr_longitude",
            "spearman_corr_latitude",
            "spearman_pvalue_longitude",
            "spearman_pvalue_latitude",
            "pearson_corr_longitude",
            "pearson_corr_latitude",
            "pearson_pvalue_longitude",
            "pearson_pvalue_latitude",
            "mad_haversine",
            "coefficient_of_variation",
            "interquartile_range",
            "percentile_25",
            "percentile_50",
            "percentile_75",
            "percent_within_20km",
            "percent_within_50km",
            "percent_within_75km",
            "mean_absolute_z_score",
        ]

        # Creating the dictionary
        metrics = dict(zip(keys, values))
        return metrics

    def get_all_stats(
        self, predictions, ground_truth, mad, coefficient_of_variation, within_threshold
    ):
        rmse = calculate_rmse(predictions, ground_truth)
        haversine_errors = self.plotting.processor.haversine_distance(
            ground_truth, predictions
        )
        mean_dist = np.mean(haversine_errors)
        median_dist = np.median(haversine_errors)
        std_dist = np.std(haversine_errors)

        (
            spearman_corr_x,
            spearman_corr_y,
            spearman_p_value_x,
            spearman_p_value_y,
        ) = self.get_correlation_coef(predictions, ground_truth, spearmanr)
        (
            pearson_corr_x,
            pearson_corr_y,
            pearson_p_value_x,
            pearson_p_value_y,
        ) = self.get_correlation_coef(predictions, ground_truth, pearsonr)

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

        # 0 is best, negative means overestimations, positive means
        # underestimations
        ks, pval, skew = kstest(ground_truth, predictions)

        return (
            z_scores,
            [
                rmse,
                mean_dist,
                median_dist,
                std_dist,
                ks,
                pval,
                skew,
                rho,
                rho_p,
                spearman_corr_x,
                spearman_corr_y,
                spearman_p_value_x,
                spearman_p_value_y,
                pearson_corr_x,
                pearson_corr_y,
                pearson_p_value_x,
                pearson_p_value_y,
                haversine_mad,
                cv,
                iqr,
                percentiles[0],
                percentiles[1],
                percentiles[2],
                percentage_within_20km,
                percentage_within_50km,
                percentage_within_75km,
                mean_absolute_z_score,
            ],
            haversine_errors,
        )

    def print_stats_to_logger(self, metrics):
        self.logger.info(f"Validation Haversine Error (km) = {metrics['mean_dist']}")
        self.logger.info(f"Median Validation Error (km) = {metrics['median_dist']}")
        self.logger.info(
            f"Standard deviation for Haversine Error (km) = {metrics['stdev_dist']}"
        )

        self.logger.info(
            f"Root Mean Squared Error (km) = {metrics['root_mean_squared_error']}"
        )
        self.logger.info(
            f"Median Absolute Deviation of Prediction Error (km) = {metrics['mad_haversine']}"
        )
        self.logger.info(
            f"Coeffiecient of Variation for Prediction Error = {metrics['coefficient_of_variation']}"
        )
        self.logger.info(
            f"Interquartile Range of Prediction Error (km) = {metrics['interquartile_range']}"
        )

        for perc in [25, 50, 75]:
            p = f"percentile_{perc}"
            self.logger.info(
                f"{perc} percentile of prediction error (km) = {metrics[p]}"
            )

        for perc in [20, 50, 75]:
            p = f"percent_within_{perc}km"

            self.logger.info(
                f"Percentage of samples with error within {perc} km = {metrics[p]}"
            )

        self.logger.info(
            f"Mean Absolute Z-scores of Prediction Error (km) = {metrics['mean_absolute_z_score']}"
        )

        self.logger.info(
            f"Spearman's Correlation Coefficient for Longitude = {metrics['spearman_corr_longitude']}, P-value = {metrics['spearman_pvalue_longitude']}"
        )
        self.logger.info(
            f"Spearman's Correlation Coefficient for Latitude = {metrics['spearman_corr_latitude']}, P-value = {metrics['spearman_pvalue_latitude']}"
        )
        self.logger.info(
            f"Pearson's Correlation Coefficient for Longitude = {metrics['pearson_corr_longitude']}, P-value = {metrics['pearson_pvalue_longitude']}"
        )
        self.logger.info(
            f"Pearson's Correlation Coefficient for Latitude = {metrics['pearson_corr_latitude']}, P-value = {metrics['pearson_pvalue_latitude']}"
        )

        # 0 is best, positive means more undeerestimations
        # negative means more overestimations.
        self.logger.info(f"Skewness = {metrics['skewness']}")

        # Goodness of fit test.
        # Small P-value means poor fit.
        # I.e., significantly deviates from reference distribution.
        self.logger.info(
            f"Kolmogorov-Smirnov Test = {metrics['kolmogorov_smirnov']}, P-value = {metrics['kolmogorov_smirnov_pval']}"
        )

    def get_correlation_coef(self, predictions, ground_truth, corr_func):
        corr_x, p_value_x = corr_func(predictions[:, 0], ground_truth[:, 0])
        corr_y, p_value_y = corr_func(predictions[:, 1], ground_truth[:, 1])
        return corr_x, corr_y, p_value_x, p_value_y

    def model_predict(self, model, data_loader, is_val=True):
        model.eval()
        predictions = []
        ground_truth = []

        with torch.no_grad():
            if is_val:
                for data, target, _ in data_loader:
                    data = torch.tensor(data, dtype=self.dtype).to(self.device)
                    target = torch.tensor(target, dtype=self.dtype).to(self.device)
                    output = model(data)
                    predictions.append(output.cpu().numpy())
                    ground_truth.append(target.cpu().numpy())
            else:
                for data in data_loader:
                    data = torch.tensor(data, dtype=self.dtype).to(self.device)
                    output = model(data)
                    predictions.append(output.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        if not is_val:
            predictions = self.ds.norm.inverse_transform(predictions)
            geo_coords_is_valid(predictions)

        if is_val:
            ground_truth = np.concatenate(ground_truth, axis=0)
            return predictions, ground_truth
        return predictions

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

                # Train the model
                (trained_model, train_losses, val_losses, centroids) = self.train_model(
                    train_loader,
                    val_loader,
                    model,
                    optimizer,
                    trial=None,
                    objective_mode=False,
                    do_bootstrap=False,
                )

            else:
                (
                    trained_model,
                    _,
                    val_losses,
                    __,
                    ___,
                    ____,
                    centroids,
                ) = self.train_rf(best_params, objective_mode=False)

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

    def write_pred_locations(self, pred_locations, pred_indices, filename):
        """write predicted locations to file."""
        if self.args.verbose >= 1:
            self.logger.info("Writing predicted coordinates to dataframe.")

        pred_locations_df = pd.DataFrame(pred_locations, columns=["x", "y"])
        pred_locations_df["sampleID"] = self.ds.all_samples[pred_indices]
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
            self.logger.warning("Optuna parameter search is not enabled.")
            return None

        if self.args.verbose >= 1:
            self.logger.info("Starting Optuna parameter search.")

        opt = Optimize(
            self.ds.train_loader,
            self.ds.val_loader,
            self.ds.test_loader,
            self.ds.samples_weight,
            self.ds.densities,
            self.ds.weighted_sampler,
            self.device,
            self.args,
            self.ds,
            show_progress_bar=False,
            n_startup_trials=10,
            dtype=self.dtype,
        )

        gb = self.args.use_gradient_boosting
        func = self.train_rf if gb else self.train_model
        best_trial, study = opt.perform_optuna_optimization(ModelClass, func)
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
        if not self.args.do_bootstrap:
            self.logger.warning("Bootstrap training is not enabled.")
            return

        if self.args.verbose >= 1:
            self.logger.info("Starting bootstrap training.")

        self.boot = Bootstrap(
            self.ds.train_loader,
            self.ds.val_loader,
            self.ds.test_loader,
            self.ds.indices["val_indices"],
            self.ds.indices["test_indices"],
            self.ds.sample_data,
            self.ds.samples,
            self.args,
            self.ds,
            best_params,
            self.ds.weighted_sampler,
            self.device,
        )

        self.boot.perform_bootstrap_training(
            self.train_model,
            self.predict_locations,
            self.write_pred_locations,
            self.make_unseen_predictions,
            ModelClass,
        )

        if self.args.verbose >= 1:
            self.logger.info("Bootstrap training completed!")

    def evaluate_and_save_results(
        self,
        model,
        train_losses,
        val_losses,
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
            centroids (np.ndarray): Centroids if using synthetic resampling with 'kerneldensity', 'none', or 'kmeans' options.; otherwise None. Defaults to None.
            use_rf (bool): Whether to use RandomForest model. If False, uses deep learning model instead. Defaults to False (deep learning model).
        """
        if self.args.verbose >= 1:
            self.logger.info(f"Evaluating the model on the {dataset} set.")

        if dataset not in {"val", "test"}:
            self.logger.error(
                "Only 'val' or 'test' are supported for the 'dataset' option."
            )
            raise ValueError(
                "Only 'val' or 'test' are supported for the 'dataset' option."
            )

        outdir = self.args.output_dir
        prefix = self.args.prefix

        middir = dataset
        if dataset.startswith("val") or dataset == "validation":
            middir = "validation"
            loader = self.ds.val_loader
        elif dataset == "test":
            loader = self.ds.test_loader
        else:
            msg = f"Invalid dataset provided. Expected 'val' or 'test', but got: {dataset}."
            self.logger.error(msg)
            raise ValueError(msg)

        y_train = self.ds.train_loader.dataset.labels.numpy()
        X_train = self.ds.train_loader.dataset.features.numpy()
        y_train = self.ds.norm.inverse_transform(y_train)

        if centroids is not None:
            centroids = self.ds.norm.inverse_transform(centroids)

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
            return_truths=True,
            use_rf=use_rf,
            dataset=dataset,
        )

        train_preds, train_metrics, y_train = self.predict_locations(
            model,
            self.train_loader_interp,
            None,  # Don't make errordist plot for train data.
            return_truths=True,
            use_rf=use_rf,
            log_metrics=False,
            is_train=True,
        )

        geo_coords_is_valid(val_preds)
        geo_coords_is_valid(y_true)
        geo_coords_is_valid(train_preds)
        geo_coords_is_valid(y_train)

        # Save validation results to file
        val_metric_outfile = os.path.join(
            outdir, middir, f"{prefix}_{middir}_metrics.json"
        )

        val_preds_outfile = os.path.join(
            outdir, middir, f"{prefix}_{middir}_predictions.txt"
        )

        train_metric_outfile = os.path.join(
            outdir, "training", f"{prefix}_train_metrics.json"
        )

        val_preds_df = self.write_pred_locations(
            val_preds, self.ds.indices[f"{dataset}_indices"], val_preds_outfile
        )

        with open(val_metric_outfile, "w") as fout:
            json.dump({k: v for k, v in val_metrics.items()}, fout, indent=2)
        with open(train_metric_outfile, "w") as fout:
            json.dump(
                {
                    k: v
                    for k, v in train_metrics.items()
                    if not isinstance(v, np.ndarray)
                },
                fout,
                indent=2,
            )

        if self.args.verbose >= 1:
            self.logger.info("Validation metrics saved.")

        if self.args.verbose >= 1:
            self.logger.info("Training metrics saved.")

        if dataset.startswith("val"):
            # Save training and validation losses to file
            train_outfile = os.path.join(
                outdir, "training", f"{prefix}_train_{dataset}_results.json"
            )
            training_results = {"train_losses": train_losses, f"val_losses": val_losses}
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

        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(val_preds, torch.Tensor):
            val_preds = val_preds.cpu().numpy()

        # Plot training history
        if not use_rf:
            self.plotting.plot_history(train_losses, val_losses)
        self.plotting.plot_geographic_error_distribution(
            y_true,
            val_preds,
            dataset,
            buffer=self.args.bbox_buffer,
            marker_scale_factor=self.args.sample_point_scale,
            min_colorscale=self.args.min_colorscale,
            max_colorscale=self.args.max_colorscale,
            n_contour_levels=self.args.n_contour_levels,
            centroids=centroids,
        )

        # self.plotting.plot_scatter_samples_map(y_train, y_true, dataset)

        self.plotting.polynomial_regression_plot(
            y_true, val_preds, dataset, dtype=self.dtype
        )
        self.plotting.polynomial_regression_plot(
            self.train_loader_interp.dataset.labels.cpu().numpy(),
            train_preds,
            "train",
            dtype=self.dtype,
        )

    def make_unseen_predictions(
        self, model, device, use_rf=False, col_indices=None, boot_rep=None
    ):
        """Predictions on data without known targets."""

        if self.args.verbose >= 1 and boot_rep is None:
            self.logger.info("Making predictions on unseen data...")

        outdir = self.args.output_dir
        prefix = self.args.prefix

        X_pred = self.ds.data["X_pred"].copy()
        if col_indices is not None:
            if boot_rep is None:
                msg = "'boot_rep' must be provided if 'col_indices' is defined."
                self.logger.error(msg)
                raise TypeError(msg)
            X_pred = X_pred[:, col_indices]

        if not use_rf:
            dtype = self.dtype

            # Convert X_pred to a PyTorch tensor and move it to the correct
            # device (GPU or CPU)
            pred_tensor = torch.tensor(X_pred, dtype=dtype).to(device)
            dataset = UnlabeledDataset(pred_tensor)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.args.batch_size, shuffle=False
            )

            # Ensures BatchNorm and Dropout layers behave correctly.
            model.eval()

            predictions = []
            with torch.no_grad():
                for data in data_loader:
                    data = data.to(device, dtype=dtype)
                    output = model(data)
                    predictions.append(output.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            pred_locations = self.ds.norm.inverse_transform(predictions)

        else:
            pred_locations_scaled = model.predict(X_pred)
            pred_locations = self.ds.norm.inverse_transform(pred_locations_scaled)

        pth = Path(outdir)
        if col_indices is None:
            basedir = "predictions"
        else:
            basedir = "bootstrap_predictions"
            prefix += f"_bootrep{boot_rep}"

        pth = pth / basedir / "unknown"
        pth.mkdir(exist_ok=True, parents=True)
        pred_outfile = pth / f"{prefix}_unknown_predictions.csv"

        if boot_rep is not None:
            return pred_locations, pred_outfile
        else:
            real_preds = self.write_pred_locations(
                pred_locations, self.ds.pred_indices, pred_outfile
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
            self.ds.define_params(self.args)
            best_params = self.ds.params

            if self.args.criterion == "drms":
                self.criterion = WeightedDRMSLoss()
            elif self.args.criterion == "rmse":
                self.criterion = weighted_rmse_loss
            elif self.args.criterion == "huber":
                self.criterion = WeightedHuberLoss(delta=0.5, smoothing_factor=0.1)
            else:
                msg = f"Invalid '--criterion' argument provided. Expected one of 'drms', 'rmse', or 'huber', but got: {self.args.criterion}"
                self.logger.error(msg)
                raise ValueError(msg)

            modelclass = "GB" if self.args.use_gradient_boosting else MLPRegressor

            # Parameter optimization with Optuna
            if self.args.do_gridsearch:
                if self.args.load_best_params is not None:
                    self.logger.warning(
                        "--load_best_params was specified; skipping paramter optimization and loading best parameeters."
                    )
                else:
                    best_params = self.optimize_parameters(ModelClass=modelclass)
                    self.ds.params = best_params
                    # Add only new keys from data_structure.params to
                    # best_params
                    for key, value in self.ds.params.items():
                        if key not in best_params:
                            best_params[key] = value
                    if self.args.verbose >= 1:
                        self.logger.info(f"Best found parameters: {best_params}")

            if self.args.load_best_params is not None:
                best_params = self.load_best_params(self.args.load_best_params)
                self.ds.params = best_params

                # Add only new keys from data_structure.params to best_params
                for key, value in self.ds.params.items():
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
                self.ds.train_loader,
                self.ds.val_loader,
                device,
                best_params,
                modelclass,
            )

            use_rf = True if modelclass in ["RF", "GB"] else False
            self.evaluate_and_save_results(
                best_model,
                train_losses,
                val_losses,
                dataset="val",
                centroids=centroids,
                use_rf=use_rf,
            )

            self.evaluate_and_save_results(
                best_model,
                train_losses,
                val_losses,
                dataset="test",
                centroids=centroids,
                use_rf=use_rf,
            )

            real_preds = self.make_unseen_predictions(best_model, device, use_rf)

            model_out = Path(outdir) / "models" / f"{prefix}_trained_model.pt"

            if self.args.verbose >= 1:
                self.logger.info("Process completed successfully!.")
                self.logger.info(f"Saving model to: {model_out}")

            if not use_rf:
                self.save_model(best_model, model_out)

            exe_pth = Path(self.args.output_dir, "benchmarking")
            exe_pth = exe_pth / f"{self.args.prefix}_execution_times.csv"
            save_execution_times(exe_pth)
            execution_times.clear()

        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()
            raise e

        self.logger.info("GeoGenIE execution succesfully completed!")
