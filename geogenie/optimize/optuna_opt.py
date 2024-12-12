import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from optuna import create_study, pruners, samplers
from optuna.logging import (
    disable_default_handler,
    enable_default_handler,
    enable_propagation,
)
from torch import optim
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.interpolate import run_genotype_interpolator
from geogenie.utils.callbacks import callback_init
from geogenie.utils.data import CustomDataset
from geogenie.utils.loss import WeightedDRMSLoss, WeightedHuberLoss, weighted_rmse_loss


class Optimize:
    """A class designed to handle the optimization of machine learning models.

    This class facilitates the process of training, validating, and testing machine learning models. It manages data loaders for different datasets, sample weights, and various parameters for model training and optimization. Additionally, it integrates functionalities for plotting  and logging the progress and results of the optimization process.

    Attributes:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        sample_weights (numpy.ndarray): Array of sample weights.
        device (str): The device (e.g., 'cpu', 'cuda') used for training.
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Patience for early stopping.
        prefix (str): Prefix used for naming output files.
        output_dir (str): Directory for saving output files.
        sqldb (str): SQL database path used for storing trial data.
        n_trials (int): Number of trials for optimization.
        n_jobs (int): Number of jobs to run in parallel.
        args (Namespace): Arguments provided for model configurations.
        show_progress_bar (bool): Flag to show or hide the progress bar during optimization.
        n_startup_trials (int): Number of initial trials to perform before applying pruning logic.
        verbose (int): Verbosity level.
        logger (Logger): Logger for logging information.
        plotting (PlotGenIE): Plotting utility for generating plots.
        cv_results (DataFrame): DataFrame to store cross-validation results.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        sample_weights,
        device,
        args,
        ds,
        show_progress_bar=False,
        n_startup_trials=10,
        dtype=torch.float32,
    ):
        """Initialize the Optimize class.

        This class is designed to handle the optimization of machine learning models. It manages data loaders for different datasets, sample weights, and various parameters for model training and optimization. Additionally, it integrates functionalities for plotting and logging the progress and results of the optimization process.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the testing dataset.
            sample_weights (numpy.ndarray): Array of sample weights.
            device (str): The device (e.g., 'cpu', 'cuda') used for training.
            args (Namespace): Arguments provided for model configurations.
            show_progress_bar (bool, optional): Flag to show or hide the progress bar. Defaults to False.
            n_startup_trials (int, optional): Number of initial trials. Defaults to 10.
            dtype (torch.dtype): PyTorch data type to use. Defaults to torch.float32.
        """

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.sample_weights = sample_weights
        self.device = device
        self.args = args
        self.ds = ds
        self.show_progress_bar = show_progress_bar
        self.n_startup_trials = n_startup_trials
        self.verbose = args.verbose
        self.sqldb = args.sqldb
        self.prefix = args.prefix
        self.n_trials = args.n_iter
        self.n_jobs = args.n_jobs
        self.output_dir = args.output_dir
        self.dtype = dtype

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            args.output_dir,
            args.prefix,
            args.basemap_fips,
            args.highlight_basemap_counties,
            args.shapefile,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
            filetype=args.filetype,
            dpi=args.plot_dpi,
            remove_splines=self.args.remove_splines,
        )

        self.cv_results = pd.DataFrame(
            columns=["trial", "average_loss", "std_dev"],
        )

    def map_sampler_indices(self, full_sampler_indices, subset_indices):
        """Map subset indices to the corresponding indices in the full dataset sampler.

        This method maps the indices used in the full dataset sampler to the indices of the desired subset. It is used to ensure that the subset sampler uses the correct indices from the full dataset.

        Args:
            full_sampler_indices (list): The indices used in the full dataset sampler.
            subset_indices (list): The indices of the desired subset.

        Returns:
            list: Mapped indices for the subset sampler.
        """
        # Create a mapping from full dataset indices to subset indices
        index_mapping = {full_index: i for i, full_index in enumerate(subset_indices)}

        # Map the sampler indices to the subset indices
        mapped_indices = [
            index_mapping.get(full_index)
            for full_index in full_sampler_indices
            if full_index in index_mapping
        ]

        return mapped_indices

    def objective_function(self, trial, ModelClass, train_func):
        """Optuna hyperparameter tuning.

        This method defines the objective function for Optuna hyperparameter tuning. It trains the model using the provided hyperparameters and returns the loss value.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            ModelClass (MLPRegressor or XGBoost): Model class to use.
            train_func (callable): Function to train model.

        Returns:
            float: Loss value.

        Raises:
            optuna.exceptions.TrialPruned: If the trial is pruned.
            ValueError: If an invalid criterion is provided.
        """
        if ModelClass == "GB":
            self.model_type = "GB"
            param_dict = self.set_gb_param_grid(trial)

        else:
            # Optuna hyperparameters
            self.model_type = "DL"
            param_dict = self.set_param_grid(trial)

        if ModelClass == "GB":
            trained_model, val_loss = self.run_rf_training(
                trial, param_dict, train_func
            )

            if np.isnan(val_loss):
                raise optuna.exceptions.TrialPruned()
            if val_loss is None:
                raise optuna.exceptions.TrialPruned()

        else:
            # Model, loss, and optimizer
            trained_model = self.run_training(trial, ModelClass, train_func, param_dict)

        if trained_model is None:
            raise optuna.exceptions.TrialPruned()

        if self.args.criterion == "drms":
            criterion = WeightedDRMSLoss()
        elif self.args.criterion == "rmse":
            criterion = weighted_rmse_loss
        elif self.args.criterion == "huber":
            criterion = WeightedHuberLoss(delta=0.5, smoothing_factor=0.1)
        else:
            msg = f"Invalid '--criterion' argument provided. Expected one of 'drms', 'rmse', or 'huber', but got: {self.args.criterion}"
            self.logger.error(msg)
            raise ValueError(msg)

        loss = self.evaluate_model(self.test_loader, trained_model, criterion)
        return loss

    def extract_features_labels(self, train_subset):
        """Extract features and labels from a subset of the training dataset.

        This method extracts features and labels from a subset of the training dataset.

        Args:
            train_subset (list): List of training data samples.

        Returns:
            tuple: Numpy arrays of features and labels.
        """
        subset_features = []
        subset_labels = []

        # Iterate over the subset and extract features and labels
        for data in train_subset:
            features, labels, _ = data
            subset_features.append(features.numpy().tolist())
            subset_labels.append(labels.numpy().tolist())
        subset_features = np.array(subset_features)
        subset_labels = np.array(subset_labels)
        return subset_features, subset_labels

    def run_rf_training(self, trial, param_dict, train_func):
        """Run random forest training.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            param_dict (dict): Dictionary of hyperparameters.
            train_func (callable): Function to train model.

        Returns:
            tuple: Trained model and validation loss.
        """
        return train_func(param_dict, objective_mode=True)

    def run_training(self, trial, ModelClass, train_func, param_dict):
        """Run deep learning model training.

        This method trains a deep learning model using the provided hyperparameters.

        Args:
            trial (optuna.Trial): Current Optuna trial.
            ModelClass (MLPRegressor or XGBoost): Model class to use.
            train_func (callable): Function to train model.
            param_dict (dict): Dictionary of hyperparameters.

        Returns:
            ModelClass: Trained model.
        """
        model = ModelClass(
            input_size=self.dataset.n_features,
            width=param_dict["width"],
            nlayers=param_dict["nlayers"],
            dropout_prop=param_dict["dropout_prop"],
            device=self.device,
            output_width=self.dataset.n_labels,
            dtype=self.dtype,
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=param_dict["lr"],
            weight_decay=param_dict["l2_weight"],
        )

        early_stop, lr_scheduler = callback_init(optimizer, self.args, trial=trial)

        # Train model
        trained_model, _ = train_func(
            self.train_loader,
            self.val_loader,
            model,
            optimizer,
            trial=trial,
            objective_mode=True,
            early_stopping=early_stop,
            lr_scheduler=lr_scheduler,
        )

        return trained_model

    def set_gb_param_grid(self, trial):
        """Function to set the parameter grid for XGBoost.

        Args:
            trial (optuna.Trial): Current Optuna trial.

        Returns:
            dict: Dictionary of hyperparameters.
        """
        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.5, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.05)
        max_depth = trial.suggest_int("max_depth", 1, 6)
        min_child_weight = trial.suggest_int("min_child_weight", 0, 10)
        reg_alpha = trial.suggest_int("reg_alpha", 0, 10)
        reg_lambda = trial.suggest_int("reg_lambda", 0, 10)
        gamma = trial.suggest_int("gamma", 0, 10)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0, step=0.01)
        colsample_bylevel = trial.suggest_float(
            "colsample_bylevel", 0.1, 1.0, step=0.01
        )
        colsample_bynode = trial.suggest_float("colsample_bynode", 0.1, 1.0, step=0.01)
        boosting = trial.suggest_categorical("boosting", ["gbtree", "gblinear", "dart"])

        tree_method = trial.suggest_categorical(
            "tree_method", ["exact", "approx", "hist"]
        )

        objective_list = ["reg:squarederror", "reg:absoluteerror"]
        objective = trial.suggest_categorical("objective", objective_list)

        return {
            "tree_method": tree_method,
            "boosting": boosting,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "gamma": gamma,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "objective": objective,
        }

    def set_param_grid(self, trial):
        """Function to set the parameter grid for deep learning models.

        Args:
            trial (optuna.Trial): Current Optuna trial.

        Returns:
            dict: Dictionary of hyperparameters.
        """
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        l2_weight = trial.suggest_float("l2_weight", 1e-5, 1e-1, log=True)
        width = trial.suggest_int(
            "width", 8, self.train_loader.dataset.tensors[0].shape[1] - 1
        )
        width_factor = 1.0
        nlayers = trial.suggest_int("nlayers", 2, 20)
        dropout_prop = trial.suggest_float("dropout_prop", 0.0, 0.5)

        return {
            "lr": lr,
            "l2_weight": l2_weight,
            "width": width,
            "width_factor": width_factor,
            "nlayers": nlayers,
            "dropout_prop": dropout_prop,
        }

    def evaluate_model(self, test_loader, model, criterion):
        """Evaluate the model using the test dataset.

        This method evaluates the model using the test dataset and returns the loss value.

        Args:
            test_loader (DataLoader): DataLoader for the test dataset.
            model (MLPRegressor or XGBoost): Trained model.
            criterion (callable): Loss function.

        Returns:
            float: Loss value.
        """
        if self.model_type == "DL":
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 4:
                        inputs, labels, sample_weights, _ = batch
                    elif len(batch) == 3:
                        inputs, labels, sample_weights = batch
                    else:
                        inputs, labels = batch
                        sample_weights = None

                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if sample_weights is not None:
                        sample_weights = sample_weights.to(self.device)
                    else:
                        sample_weights = torch.ones(
                            len(labels), dtype=self.dtype, device=self.device
                        )

                    outputs = model(inputs)
                    loss = criterion(outputs, labels, sample_weights)

                    total_loss += loss.item()

            return total_loss / len(test_loader)

        else:
            X_true = test_loader.dataset.features.numpy()
            y_true = test_loader.dataset.labels.numpy()
            y_pred = model.predict(X_true)

            total_haversine = np.mean(
                self.plotting.processor.haversine_distance(y_true, y_pred)
            )
            return total_haversine

    def perform_optuna_optimization(self, ModelClass, train_func):
        """Perform parameter optimization using Optuna.

        This method performs parameter optimization using Optuna. It trains the model using the provided hyperparameters and returns the best trial and the Optuna study object.

        Args:
            ModelClass (MLPRegressor or XGBoost): Model class to use.
            train_func (callable): Function to train model.

        Returns:
            tuple: Best trial and the Optuna study object.
        """

        # Enable log propagation to the root logger
        enable_propagation()

        # Disable Optuna's default handler to avoid double logging
        disable_default_handler()

        if self.args.oversample_method != "none":
            self.train_loader, _, __, ___, ____ = run_genotype_interpolator(
                self.train_loader, self.args, self.ds, self.dtype, self.plotting
            )

        self.dataset = self.train_loader.dataset

        # Define the objective function for Optuna
        def objective(trial):
            return self.objective_function(trial, ModelClass, train_func)

        # Optuna Optimization setup
        sampler = samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=24,
        )
        pruner = pruners.MedianPruner()

        if self.sqldb is None:
            storage_path = None
        else:
            Path(self.sqldb).mkdir(parents=True, exist_ok=True)
            storage_path = f"sqlite:///{self.sqldb}/{self.prefix}_optuna.db"
            if self.verbose >= 1:
                self.logger.info(f"Writing Optuna data to database: {storage_path}")

        if self.verbose < 1:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            storage=storage_path,
            load_if_exists=True,
            study_name=f"{self.prefix}_torch_study",
        )

        if self.verbose >= 1:
            self.logger.info("Beginning parameter search...")

        start_time = time.time()

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress_bar,
        )

        end_time = time.time()
        total_time = end_time - start_time

        outdir = Path(self.output_dir)
        cv_outfile = outdir / "optimize" / f"{self.prefix}_cv_results.csv"
        self.cv_results.to_csv(cv_outfile, header=True, index=False)

        if self.verbose >= 1:
            self.logger.info(f"Finished parameter search in {total_time} seconds")

        return study.best_trial, study

    def process_optuna_results(self, study, best_trial):
        """Process and save the results of the Optuna optimization study.

        This method processes the results of the Optuna optimization study and saves the best parameters to a file. It also generates and saves plots and output files.

        Args:
            study (optuna.Study): The Optuna study object containing the results of the hyperparameter optimization.
            best_trial (optuna.Trial): The best trial from the Optuna study.
        """
        # Extract and print the best parameters
        best_params = best_trial.params

        if self.verbose >= 1:
            self.logger.info(f"Best trial parameters: {best_params}")

        outdir = Path(self.output_dir)
        fn = outdir / "optimize" / f"{self.prefix}_best_params.txt"
        jfn = outdir / "optimize" / f"{self.prefix}_best_params.json"

        # Save the best parameters to a file
        with open(fn, "w") as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")

        with open(jfn, "w") as f:
            json.dump(best_params, f, indent=2)

        # Generate and save plots and output files.
        self.plotting.make_optuna_plots(study)
        self.write_optuna_study_details(study)

        enable_default_handler()

    def write_optuna_study_details(self, study):
        """Write Optuna study to file.

        This method writes the results of the Optuna study to files for further analysis.

        Args:
            study (optuna.Study): The Optuna study object containing the results of the hyperparameter optimization.
        """

        if self.verbose >= 2 or self.args.debug:
            self.logger.info("Writing parameter optimizations to file...")

        outdir = Path(self.output_dir, "optimize")

        df = study.trials_dataframe()
        df.to_csv(outdir / f"{self.prefix}_trials_df.csv", header=True)

        with open(outdir / f"{self.prefix}_sampler.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)

        with open(outdir / f"{self.prefix}_best_score.txt", "w") as fout:
            fout.write(str(study.best_value))

        with open(outdir / f"{self.prefix}_best_params.pkl", "wb") as fout:
            pickle.dump(study.best_params, fout)

        with open(outdir / f"{self.prefix}_best_trials.pkl", "wb") as fout:
            pickle.dump(study.best_trials, fout)

        with open(outdir / f"{self.prefix}_best_overall_trial.pkl", "wb") as fout:
            pickle.dump(study.best_trial, fout)

        with open(outdir / f"{self.prefix}_all_trials.pkl", "wb") as fout:
            pickle.dump(study.trials, fout)
