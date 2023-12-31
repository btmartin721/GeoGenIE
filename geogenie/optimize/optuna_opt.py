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
from sklearn.model_selection import StratifiedKFold
from torch import optim
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.loss import (
    WeightedRMSELoss,
    WeightedHaversineLoss,
    euclidean_distance_loss,
)
from geogenie.utils.utils import CustomDataset
from geogenie.utils.scorers import haversine_distances_agg


class Optimize:
    """
    A class designed to handle the optimization of machine learning models.

    This class facilitates the process of training, validating, and testing machine learning models. It manages data loaders for different datasets, sample weights, and various parameters for model training and optimization. Additionally, it integrates functionalities for plotting  and logging the progress and results of the optimization process.

    Attributes:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        sample_weights (numpy.ndarray): Array of sample weights.
        weighted_sampler (Sampler): Sampler that applies sample weights.
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

    Args:
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the testing dataset.
        sample_weights (numpy.ndarray): Array of sample weights.
        weighted_sampler (Sampler): Sampler that applies sample weights.
        device (str): The device (e.g., 'cpu', 'cuda') used for training.
        args (Namespace): Arguments provided for model configurations.
        show_progress_bar (bool, optional): Flag to show or hide the progress bar. Defaults to False.
        n_startup_trials (int, optional): Number of initial trials. Defaults to 10.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        sample_weights,
        weighted_sampler,
        device,
        args,
        show_progress_bar=False,
        n_startup_trials=10,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.sample_weights = sample_weights
        self.weighted_sampler = weighted_sampler
        self.device = device
        self.args = args
        self.show_progress_bar = show_progress_bar
        self.n_startup_trials = n_startup_trials
        self.verbose = args.verbose
        self.sqldb = args.sqldb
        self.prefix = args.prefix
        self.n_trials = args.n_iter
        self.n_jobs = args.n_jobs
        self.output_dir = args.output_dir

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            args.output_dir,
            args.prefix,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
            filetype=args.filetype,
            dpi=args.plot_dpi,
        )

        self.cv_results = pd.DataFrame(
            columns=["trial", "average_loss", "std_dev"],
        )

    def map_sampler_indices(self, full_sampler_indices, subset_indices):
        """
        Map subset indices to the corresponding indices in the full dataset sampler.

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

    def objective_function(self, trial, criterion, ModelClass, train_func):
        """Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): Current Optuna trial.

        Returns:
            float: Loss value.
        """
        if ModelClass == "RF":
            # RF Hyperparameters
            self.model_type = "RF"
            param_dict = self.set_rf_param_grid(trial, self.dataset.features.shape[0])

        elif ModelClass == "GB":
            self.model_type = "GB"
            param_dict = self.set_gb_param_grid(trial)

        else:
            # Optuna hyperparameters
            self.model_type = "DL"
            param_dict = self.set_param_grid(trial)

        if ModelClass in ["RF", "GB"]:
            trained_model, val_loss = self.run_rf_training(
                trial, param_dict, train_func
            )
        else:
            criterion = euclidean_distance_loss

            # Model, loss, and optimizer
            trained_model, val_loss = self.run_training(
                trial, criterion, ModelClass, train_func, param_dict
            )

        if trained_model is None:
            raise optuna.exceptions.TrialPruned()

        _, haversine_error = self.evaluate_model(
            self.val_loader, trained_model, euclidean_distance_loss
        )

        if np.isnan(val_loss):
            raise optuna.exceptions.TrialPruned()
        if val_loss is None:
            raise optuna.exceptions.TrialPruned()

        return haversine_error

    def extract_features_labels(self, train_subset):
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
        return train_func(param_dict, objective_mode=True)

    def run_training(self, trial, criterion, ModelClass, train_func, param_dict):
        model = ModelClass(
            input_size=self.dataset.features.shape[1],
            width=param_dict["width"],
            nlayers=param_dict["nlayers"],
            dropout_prop=param_dict["dropout_prop"],
            device=self.device,
            factor=param_dict["width_factor"],
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=param_dict["lr"],
            weight_decay=param_dict["l2_weight"],
        )

        # Train model
        trained_model = train_func(
            self.train_loader,
            self.val_loader,
            model,
            criterion,
            optimizer,
            trial=trial,
            objective_mode=True,
        )

        return trained_model

    def reload_subsets(
        self,
        use_weighted,
        kwargs,
        val_subset,
        subset_features,
        subset_labels,
        weighted_sampler,
        subset_features_val=None,
        subset_labels_val=None,
    ):
        train_subset_dataset = CustomDataset(
            subset_features, subset_labels, weighted_sampler.weights
        )

        if self.model_type in ["RF", "GB"]:
            val_subset_dataset = CustomDataset(subset_features_val, subset_labels_val)

        if use_weighted in ["sampler", "both"]:
            kwargs["sampler"] = weighted_sampler
        else:
            kwargs["shuffle"] = True
        train_loader = DataLoader(train_subset_dataset, **kwargs)

        if self.model_type in ["RF", "GB"]:
            val_loader = DataLoader(
                val_subset_dataset, batch_size=kwargs["batch_size"], shuffle=False
            )
        else:
            val_loader = DataLoader(
                val_subset, batch_size=kwargs["batch_size"], shuffle=False
            )

        return train_loader, val_loader

    def set_gb_param_grid(self, trial):
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

    def set_rf_param_grid(self, trial, n_samples):
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        criterion = trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error", "friedman_mse"]
        )
        max_depth = trial.suggest_int("max_depth", 2, n_samples)
        min_samples_split = trial.suggest_float("min_samples_split", 1e-3, 1.0)
        min_samples_leaf = trial.suggest_float("min_samples_leaf", 1e-3, 1.0)
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.1, 0.3, 0.5, 0.9, None]
        )
        max_samples = trial.suggest_float("max_samples", 1e-3, 1.0)

        return {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": True,
            "oob_score": True,
            "max_samples": max_samples,
        }

    def set_param_grid(self, trial):
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

    def evaluate_model(self, val_loader, model, criterion):
        if self.model_type == "DL":
            model.eval()
            total_loss = 0.0
            total_haversine = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        inputs, labels, sample_weights = batch
                        inputs, labels, sample_weights = (
                            inputs.to(self.device),
                            labels.to(self.device),
                            sample_weights.to(self.device),
                        )

                    else:
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        sample_weights = None
                    outputs = model(inputs)
                    loss = criterion(
                        outputs,
                        labels,
                        torch.ones(len(sample_weights), dtype=torch.float32),
                    )

                    haverror = haversine_distances_agg(
                        labels.numpy(), outputs.numpy(), np.median
                    )

                    total_loss += loss.item()
                    total_haversine += haverror
            return total_loss / len(val_loader), total_haversine / len(val_loader)
        else:
            X_true = val_loader.dataset.features.numpy()
            y_true = val_loader.dataset.labels.numpy()
            y_pred = model.predict(X_true)

            total_haversine = haversine_distances_agg(y_true, y_pred, np.mean)
            return None, total_haversine

    def perform_optuna_optimization(self, criterion, ModelClass, train_func):
        """
        Perform parameter optimization using Optuna.

        Args:
            logger (logging): Logger to write output to.

        Returns:
            tuple: Best trial and the Optuna study object.
        """
        self.dataset = self.train_loader.dataset

        # Enable log propagation to the root logger
        enable_propagation()

        # Disable Optuna's default handler to avoid double logging
        disable_default_handler()

        # Define the objective function for Optuna
        def objective(trial):
            return self.objective_function(
                trial,
                criterion,
                ModelClass,
                train_func,
            )

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

        cv_outfile = os.path.join(
            self.output_dir, "optimize", f"{self.prefix}_cv_results.csv"
        )

        self.cv_results.to_csv(cv_outfile, header=True, index=False)

        if self.verbose >= 1:
            self.logger.info(f"Finished parameter search in {total_time} seconds")

        return study.best_trial, study

    def process_optuna_results(self, study, best_trial):
        """
        Process and save the results of the Optuna optimization study.

        Args:
            study (optuna.Study): The Optuna study object containing the results of the hyperparameter optimization.
            best_trial (optuna.Trial): The best trial from the Optuna study.

        Returns:
            None
        """
        # Extract and print the best parameters
        best_params = best_trial.params

        if self.verbose >= 1:
            self.logger.info(f"Best trial parameters: {best_params}")

        fn = os.path.join(
            self.output_dir,
            "optimize",
            f"{self.prefix}_best_params.txt",
        )

        jfn = os.path.join(
            self.output_dir, "optimize", f"{self.prefix}_best_params.json"
        )

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
        """Write Optuna study to file."""

        if self.verbose >= 2:
            self.logger.info("Writing parameter optimizations to file...")

        outdir = os.path.join(self.output_dir, "optimize")

        df = study.trials_dataframe()
        df.to_csv(
            os.path.join(outdir, f"{self.prefix}_trials_df.csv"),
            header=True,
        )

        with open(os.path.join(outdir, f"{self.prefix}_sampler.pkl"), "wb") as fout:
            pickle.dump(study.sampler, fout)

        with open(os.path.join(outdir, f"{self.prefix}_best_score.txt"), "w") as fout:
            fout.write(str(study.best_value))

        with open(os.path.join(outdir, f"{self.prefix}_best_params.pkl"), "wb") as fout:
            pickle.dump(study.best_params, fout)

        with open(os.path.join(outdir, f"{self.prefix}_best_trials.pkl"), "wb") as fout:
            pickle.dump(study.best_trials, fout)

        with open(
            os.path.join(outdir, f"{self.prefix}_best_overall_trial.pkl"), "wb"
        ) as fout:
            pickle.dump(study.best_trial, fout)

        with open(os.path.join(outdir, f"{self.prefix}_all_trials.pkl"), "wb") as fout:
            pickle.dump(study.trials, fout)
