import json
import logging
import os
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from geogenie.models.models import MLPRegressor
from geogenie.optimize.boostrap import Bootstrap
from geogenie.optimize.optuna_opt import Optimize
from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler
from geogenie.utils.callbacks import EarlyStopping
from geogenie.utils.data_structure import DataStructure
from geogenie.utils.loss import MultiobjectiveHaversineLoss
from geogenie.utils.scorers import haversine_distances_agg, kstest, r2_multioutput
from geogenie.utils.utils import geo_coords_is_valid
from geogenie.utils.scorers import haversine


class GeoGenIE:
    """A class for predicting geographic localities from genomic SNP data using neural networks."""

    def __init__(self, args):
        """
        Initializes the GeoGenIE class.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        self.args = args
        self.genotypes = None
        self.samples = None
        self.sample_data = None
        self.locs = None

        if self.args.gpu_number is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_number
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.device = torch.device("cpu")

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
        ]

        output_dir = self.args.output_dir
        prefix = self.args.prefix

        [
            Path(os.path.join(output_dir, d)).mkdir(exist_ok=True, parents=True)
            for d in output_dir_list
        ]

        self.logger = logging.getLogger()  # Use root logger

        self.plotting = PlotGenIE(
            self.device,
            output_dir,
            prefix,
            self.args.show_plots,
            self.args.fontsize,
        )

    def load_data(self):
        """Loads genotypes from VCF file using pysam."""
        if self.args.vcf is not None:
            self.data_structure = DataStructure(self.args.vcf)
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

    def load_model(self, input_size, ModelClass, filename, device):
        """
        Loads a model from a file.

        Args:
            input_size (int): Size of input layer.
            width (int): Number of neurons in hidden layers.
            nlayes (int): Number of hidden layers.
            dropout_prop (float): Dropout layer proportion.
            ModelClass (torch.nn.Module): The class of the model to load.
            filename (str): The path to the file from which the model will be loaded.
            device (torch.device): The device to load the model onto.

        Returns:
            torch.nn.Module: The loaded PyTorch model.
        """

        model = ModelClass(
            input_size,
            width=self.args.width,
            nlayers=self.args.nlayers,
            dropout_prop=self.args.dropout_prop,
            device=device,
            embedding_dim=self.args.embedding_dim,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
        )
        model.load_state_dict(torch.load(filename, map_location=device))
        model.to(device)
        return model

    def train_model(
        self,
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        trial,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=8,
        objective_mode=False,
        verbose=False,
        grad_clip=False,
    ):
        """
        Train the PyTorch model with given parameters.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): PyTorch model to be trained.
            trial (optuna.Trial): Optuna trial object, if applicable.
            criterion: Loss function.
            optimizer: Optimizer.
            lr_scheduler_factor (float): Factor by which the learning rate will be reduced.
            lr_scheduler_patience (int): Number of epochs with no improvement after which learning rate will be reduced.
            objective_mode (bool): Whether to return just the model for Optuna's objective function.
            grad_clip (bool): If True, does gradient clipping.

        Returns:
            Model and training statistics, or just the model if in objective mode.
        """
        if verbose >= 2:
            verbose = True
        else:
            verbose = False

        early_stopping = EarlyStopping(
            output_dir=self.args.output_dir,
            prefix=self.args.prefix,
            patience=self.args.patience,
            verbose=verbose,
            delta=0,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            verbose=verbose,
        )

        train_losses, val_losses, epoch_times = [], [], []
        rolling_window_size = 20
        total_time = 0

        epochs = self.args.max_epochs

        if verbose:
            self.logger.info("\n\n")

        if self.args.class_weights and self.args.model_type == "gcn":
            sample_weights = torch.tensor(
                self.data_structure.samples_weight,
                dtype=torch.float32,
                device=self.device,
            )

        for epoch in range(epochs):
            try:
                start_time = time.time()  # Start time for the epoch
                model.train()
                total_loss = 0.0

                if self.args.model_type == "gcn":
                    optimizer.zero_grad()
                    outputs = model(train_loader)
                    loss = criterion(
                        outputs,
                        self.data_structure.train_dataset[1],
                        sample_weight=sample_weights,
                    )
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    avg_train_loss = loss.item()

                    model.eval()
                    with torch.no_grad():
                        outputs = model(val_loader)
                        val_loss = criterion(
                            outputs,
                            self.data_structure.val_dataset[1],
                            sample_weight=sample_weights,
                        )

                        val_losses.append(val_loss.item())
                        avg_val_loss = val_loss.item()
                else:
                    for batch in train_loader:
                        if self.args.model_type == "gcn":
                            data = batch
                        else:
                            if len(batch) == 3:
                                data, targets, sample_weight = batch
                            else:
                                data, targets = batch
                        if self.args.model_type == "transformer":
                            data = data.long()

                        if self.args.model_type != "gcn":
                            data = data.to(model.device)
                            targets = targets.to(model.device)

                            if sample_weight is not None:
                                sample_weight = sample_weight.to(model.device)

                        optimizer.zero_grad()
                        outputs = model(data)
                        loss = criterion(outputs, targets, sample_weight=sample_weight)
                        loss.backward()

                        if grad_clip and self.args.model_type == "mlp":
                            # Gradient clipping.
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        total_loss += loss.item()
                    avg_train_loss = total_loss / len(train_loader)

                    train_losses.append(avg_train_loss)

                    # Validation
                    model.eval()
                    total_val_loss = 0.0
                    with torch.no_grad():
                        for batch in val_loader:
                            if self.args.model_type == "gcn":
                                data = batch
                            else:
                                if len(batch) == 3:
                                    data, targets, sample_weight = batch
                                else:
                                    data, targets = batch
                            if self.args.model_type == "transformer":
                                data = data.long()

                            if self.args.model_type != "gcn":
                                data = data.to(model.device)
                                targets = targets.to(model.device)

                                if sample_weight is not None:
                                    sample_weight = sample_weight.to(model.device)
                            outputs = model(data)
                            val_loss = criterion(
                                outputs,
                                targets,
                                sample_weight=sample_weight,
                            )
                            total_val_loss += val_loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    val_losses.append(avg_val_loss)

                if verbose:
                    self.logger.info(
                        f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {avg_train_loss[-1]:.4f} - "
                        f"Final Val Loss: {avg_val_loss[-1]:.4f}"
                    )

                end_time = time.time()
                epoch_duration = end_time - start_time
                epoch_times.append(epoch_duration)

                # Logging
                if verbose and epoch % (rolling_window_size * 2) == 0:
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
                    if verbose:
                        self.logger.info("Early stopping triggered.")
                    break

                if trial is not None and trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except Exception as e:
                self.logger.error(
                    f"Error during training at epoch {epoch}: {e}",
                )
                raise

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
            return model
        else:
            return (
                model,
                train_losses,
                val_losses,
                rolling_avgs,
                rolling_stds,
                total_time,
            )

    def compute_rolling_statistics(self, times, window_size):
        """Compute rolling average and standard deviation."""
        averages = []
        std_devs = []
        for i in range(len(times)):
            window = times[max(0, i - window_size + 1) : i + 1]
            avg = np.mean(window)
            std = np.std(window)
            averages.append(avg)
            std_devs.append(std)
        return averages, std_devs

    @staticmethod
    def euclidean_distance_loss(y_true, y_pred):
        """Custom PyTorch loss function."""
        return torch.sqrt(torch.sum((y_pred - y_true) ** 2, axis=1)).mean()

    @staticmethod
    def haversine_distance_torch(lon1, lat1, lon2, lat2):
        """
        Calculate the Haversine distance between two points on the earth in PyTorch.
        Args:
            lat1, lon1, lat2, lon2: latitude and longitude of two points in radians.
        Returns:
            Distance in kilometers.
        """
        R = 6371.0  # Earth radius in kilometers
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2.0) ** 2
            + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
        )
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        return R * c

    @staticmethod
    def haversine_loss(pred, target, eps=1e-6):
        lon1, lat1 = torch.split(pred, 1, dim=1)
        lon2, lat2 = torch.split(target, 1, dim=1)
        r = 6371  # Radius of Earth in kilometers

        phi1, phi2 = torch.deg2rad(lat1), torch.deg2rad(lat2)
        delta_phi = torch.deg2rad(lat2 - lat1)
        delta_lambda = torch.deg2rad(lon2 - lon1)

        a = (
            torch.sin(delta_phi / 2) ** 2
            + torch.cos(phi1) * torch.cos(phi2) * torch.sin(delta_lambda / 2) ** 2
        )
        a = torch.clamp(a, min=0, max=1)  # Clamp 'a' to avoid sqrt of negative number

        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a + eps))
        distance = r * c  # Compute the distance

        return torch.mean(distance)

    def predict_locations(
        self,
        model,
        data_loader,
        device,
        return_truths=False,
    ):
        """
        Predict locations using the trained model and evaluate predictions.

        Args:
            args (argparse.Namespace): argparsed arguments from command-line.
            model (torch.nn.Module): Trained PyTorch model for predictions.
            data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for prediction.
            device (torch.device): Device to run the model on ('cpu' or 'cuda').
            coord_scaler (dict): Dictionary with meanlong, meanlat, stdlong, stdlat.
            return_truths (bool): Whether to return truths as well as predictions.

        Returns:
            pandas.DataFrame: DataFrame with predicted locations and corresponding sample IDs.
        """
        model.eval()
        predictions = []
        ground_truth = []
        with torch.no_grad():
            for data, target, sample_weight in data_loader:
                if self.args.model_type == "transformer":
                    data = data.long()
                data = data.to(device)
                output = model(data)
                predictions.append(output.cpu().numpy())
                ground_truth.append(target.numpy())
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)

        def rescale_predictions(y):
            return self.data_structure.norm.inverse_transform(y)

        # Rescale predictions and ground truth to original scale
        predictions = rescale_predictions(predictions)
        ground_truth = rescale_predictions(ground_truth)

        geo_coords_is_valid(predictions)
        geo_coords_is_valid(ground_truth)

        # Evaluate predictions
        r2_lon, r2_lat = r2_multioutput(predictions, ground_truth)
        mean_dist = haversine_distances_agg(ground_truth, predictions, np.mean)
        median_dist = haversine_distances_agg(ground_truth, predictions, np.median)
        std_dist = haversine_distances_agg(ground_truth, predictions, np.std)

        # 0 is best, negative means overestimations, positive means
        # underestimations
        ks, pval, skew = kstest(ground_truth, predictions)

        self.logger.info(f"R2(x) = {r2_lon}")
        self.logger.info(f"R2(y) = {r2_lat}")
        self.logger.info(f"Validation Distance Error (km) = {mean_dist}")
        self.logger.info(f"Median Validation Error (km) = {median_dist}")
        self.logger.info(f"Standard deviation for Error (km) = {std_dist}")

        # 0 is best, positive means more undeerestimations
        # negative means more overestimations.
        self.logger.info(f"Skewness = {skew}")

        # Goodness of fit test.
        # Small P-value means poor fit.
        # I.e., significantly deviates from reference distribution.
        self.logger.info(f"Kolmogorov-Smirnov Test = {ks}, P-value = {pval}")

        # Calculate Haversine error for each pair of points
        haversine_errors = np.array(
            [
                haversine(act[0], act[1], pred[0], pred[1])
                for act, pred in zip(ground_truth, predictions)
            ]
        )

        self.plotting.plot_error_distribution(haversine_errors)

        # return the evaluation metrics along with the predictions
        metrics = {
            "r2_long": r2_lon,
            "r2_lat": r2_lat,
            "mean_dist": mean_dist,
            "median_dist": median_dist,
            "stdev_dist": std_dist,
            "ks": ks,
            "ks_pval": pval,
            "skewness": skew,
        }

        if return_truths:
            return predictions, metrics, ground_truth
        return predictions, metrics

    def plot_bootstrap_aggregates(self, df, filename, train_times):
        self.plotting.plot_bootstrap_aggregates(df, filename)

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
        self,
        train_loader,
        val_loader,
        device,
        best_params,
        ModelClass,
        criterion,
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

            # Initialize the model
            model = ModelClass(
                input_size=train_loader.dataset.features.shape[1],
                device=device,
                **best_params,
            ).to(device)

            criterion, optimizer = self.extract_best_params(best_params, model)
            train_loader = self.get_sample_weights(train_loader, best_params)

            # Train the model
            (
                trained_model,
                train_losses,
                val_losses,
                _,
                __,
                total_train_time,
            ) = self.train_model(
                train_loader,
                val_loader,
                model,
                criterion,
                optimizer,
                None,
                best_params["lr_scheduler_factor"],
                best_params["lr_scheduler_patience"],
                objective_mode=False,
                verbose=self.args.verbose,
                grad_clip=best_params["grad_clip"],
            )

            if self.args.verbose >= 1:
                self.logger.info(
                    f"Standard training completed in {total_train_time / 60} "
                    f"minutes",
                )

            return trained_model, train_losses, val_losses

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

        criterion = MultiobjectiveHaversineLoss(
            alpha=best_params["alpha"],
            beta=best_params["beta"],
            gamma=best_params["gamma"],
        )

        return criterion, optimizer

    def get_sample_weights(self, train_loader, best_params):
        if best_params["use_weighted"] in ["sampler", "loss", "both"]:
            weighted_sampler = GeographicDensitySampler(
                pd.DataFrame(train_loader.dataset.labels, columns=["x", "y"]),
                None,
                best_params["use_kmeans"],
                best_params["use_kde"],
                best_params["w_power"],
                best_params["max_clusters"],
                best_params["max_neighbors"],
            )

            sample_weights = weighted_sampler.weights

            if sample_weights is None:
                sample_weights = torch.ones(
                    train_loader.dataset.labels.shape[0], dtype=torch.float32
                )

            train_dataset = train_loader.dataset
            if best_params["use_weighted"] in ["loss", "both"]:
                train_dataset.sample_weights = sample_weights

            kwargs = {"batch_size": train_loader.batch_size}
            if best_params["use_weighted"] in ["sampler", "both"]:
                kwargs["sampler"] = weighted_sampler
            else:
                kwargs["shuffle"] = True
            train_loader = DataLoader(train_dataset, **kwargs)
        return train_loader

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

    def optimize_parameters(self, criterion, ModelClass):
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
            self.data_structure.weighted_sampler,
            self.device,
            self.args.max_epochs,
            self.args.patience,
            self.args.prefix,
            self.args.output_dir,
            self.args.sqldb,
            self.args.n_iter,
            self.args.n_jobs,
            show_progress_bar=False,
            n_startup_trials=10,
            verbose=self.args.verbose,
        )

        best_trial, study = opt.perform_optuna_optimization(
            criterion, ModelClass, self.train_model
        )
        opt.process_optuna_results(study, best_trial)

        if self.args.verbose >= 1:
            self.logger.info("Optuna optimization completed!")

        return best_trial.params

    def perform_bootstrap_training(self, criterion, ModelClass, best_params):
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
            self.args.patience,
            self.args.output_dir,
            self.args.prefix,
            self.data_structure.sample_data,
            verbose=self.args.verbose,
            show_plots=self.args.show_plots,
            fontsize=self.args.fontsize,
        )

        boot.perform_bootstrap_training(
            self.train_model,
            self.predict_locations,
            self.write_pred_locations,
            ModelClass,
            criterion,
            self.data_structure.coord_scaler,
        )

        if self.args.verbose >= 1:
            self.logger.info("Bootstrap training completed.")

    def evaluate_and_save_results(
        self, model, train_losses, val_losses, device, dataset="val"
    ):
        """
        Evaluate the model and save the results.

        Args:
            model: The trained model to evaluate.
            train_losses: List of training losses.
            val_losses: List of validation losses.
            device (str): Device to use. Either 'cpu' or 'cuda'.
            dataset (str): Whether 'val' or 'test' dataset.
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

        val_preds, val_metrics, y_true = self.predict_locations(
            model,
            loader,
            device,
            return_truths=True,
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
            json.dump(
                {k: v.astype("float64") for k, v in val_metrics.items()}, fout, indent=2
            )

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
            json.dump(training_results, fout, indent=2)

        if self.args.verbose >= 1:
            self.logger.info("Training and validation losses saved.")

        hist_outdir = os.path.join(
            outdir,
            "plots",
            f"{prefix}_train_history.png",
        )

        geo_outfile = os.path.join(
            self.args.output_dir,
            "plots",
            f"{self.args.prefix}_geographic_error_{dataset}.png",
        )
        # Plot training history
        self.plotting.plot_history(train_losses, val_losses, hist_outdir)
        self.plotting.plot_geographic_error_distribution(
            y_true,
            val_preds,
            geo_outfile,
            self.args.shapefile_url,
            buffer=0.1,
            marker_scale_factor=3,
        )

        self.plotting.plot_kde_error_regression(y_true, val_preds)

        if self.args.verbose >= 1:
            self.logger.info("Training history plotted.")

    def make_unseen_predictions(self, model, device):
        if self.args.verbose >= 1:
            # Predictions on unseen data
            self.logger.info("Making predictions on unseen data...")

        outdir = self.args.output_dir
        prefix = self.args.prefix

        if self.args.model_type == "transformer":
            dtype = torch.long
        else:
            dtype = torch.float32

        # Convert X_pred to a PyTorch tensor and move it to the correct
        # device (GPU or CPU)
        pred_tensor = torch.tensor(self.data_structure.data["X_pred"], dtype=dtype).to(
            device
        )

        with torch.no_grad():
            # Make predictions
            pred_locations_scaled = model(pred_tensor)

        pred_locations = self.data_structure.norm.inverse_transform(
            pred_locations_scaled.cpu().numpy()
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
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu_number
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

            criterion = MultiobjectiveHaversineLoss()
            modelclass = MLPRegressor

            if self.args.model_type.lower() != "mlp":
                msg = f"Invalid 'model_type' provided: {self.args.model_type}"
                self.logger.error(msg)
                raise NotImplementedError(msg)

            # Parameter optimization with Optuna
            if self.args.do_gridsearch:
                best_params = self.optimize_parameters(
                    criterion=criterion, ModelClass=modelclass
                )
                self.data_structure.params = best_params
                self.logger.info(f"Best found parameters: {best_params}")

            # Model Training
            if self.args.bootstrap:
                self.perform_bootstrap_training(
                    criterion,
                    modelclass,
                    best_params,
                )

            (
                best_model,
                train_losses,
                val_losses,
            ) = self.perform_standard_training(
                self.data_structure.train_loader,
                self.data_structure.test_loader,
                device,
                best_params,
                modelclass,
                criterion,
            )

            self.evaluate_and_save_results(
                best_model,
                train_losses,
                val_losses,
                device,
                dataset="val",
            )

            self.evaluate_and_save_results(
                best_model, train_losses, val_losses, device, dataset="test"
            )

            real_preds = self.make_unseen_predictions(best_model, device)

            model_out = os.path.join(
                outdir,
                "models",
                f"{prefix}_trained_model.pt",
            )

            if self.args.verbose >= 1:
                self.logger.info("Process completed successfully!.")
                self.logger.info(f"Saving model to: {model_out}")

            self.save_model(best_model, model_out)

        except Exception as e:
            self.logger.error(f"Unexpected error occurred: {e}")
            traceback.print_exc()
            raise
