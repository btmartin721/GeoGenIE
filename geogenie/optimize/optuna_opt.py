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
from geogenie.utils.loss import WeightedRMSELoss
from geogenie.utils.utils import CustomDataset


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
        max_epochs (int): Maximum number of epochs for training.
        patience (int): Patience for early stopping.
        prefix (str): Prefix used for naming output files.
        output_dir (str): Directory for saving output files.
        sqldb (str): SQL database path used for storing trial data.
        n_trials (int): Number of trials for optimization.
        n_jobs (int): Number of jobs to run in parallel.
        args (Namespace): Arguments provided for model configurations.
        show_progress_bar (bool, optional): Flag to show or hide the progress bar. Defaults to False.
        n_startup_trials (int, optional): Number of initial trials. Defaults to 10.
        show_plots (bool, optional): Flag to show or hide plots. Defaults to False.
        fontsize (int, optional): Font size for plots. Defaults to 18.
        filetype (str, optional): File type for saving plots. Defaults to "png".
        dpi (int, optional): Dots per inch for plot resolution. Defaults to 300.
        verbose (int, optional): Verbosity level. Defaults to 1.
    """

    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        sample_weights,
        weighted_sampler,
        device,
        max_epochs,
        patience,
        prefix,
        output_dir,
        sqldb,
        n_trials,
        n_jobs,
        args,
        show_progress_bar=False,
        n_startup_trials=10,
        show_plots=False,
        fontsize=18,
        filetype="png",
        dpi=300,
        verbose=1,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.sample_weights = sample_weights
        self.weighted_sampler = weighted_sampler
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.prefix = prefix
        self.output_dir = output_dir
        self.sqldb = sqldb
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.args = args
        self.show_progress_bar = show_progress_bar
        self.n_startup_trials = n_startup_trials
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device,
            output_dir,
            prefix,
            show_plots=show_plots,
            fontsize=fontsize,
            filetype=filetype,
            dpi=dpi,
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
            (
                param_dict,
                n_bins,
                use_weighted,
                smote_method,
                smote_neighbors,
            ) = self.set_rf_param_grid(trial, self.dataset.features.shape[0])

            use_kmeans = False
            use_kde = True
            w_power = 1.0
            max_clusters = 10
            max_neighbors = 50
            normalize = False

        elif ModelClass == "GB":
            self.model_type = "GB"
            (
                param_dict,
                n_bins,
                use_weighted,
                smote_method,
                smote_neighbors,
            ) = self.set_gb_param_grid(trial, self.dataset.features.shape[0])

            use_kmeans = False
            use_kde = True
            w_power = 1.0
            max_clusters = 10
            max_neighbors = 50
            normalize = False

        else:
            # Optuna hyperparameters
            self.model_type = "DL"
            (
                lr,
                width,
                nlayers,
                dropout_prop,
                l2_weight,
                lr_scheduler_patience,
                lr_scheduler_factor,
                width_factor,
                use_kmeans,
                use_kde,
                w_power,
                normalize,
                use_weighted,
                max_clusters,
                max_neighbors,
                grad_clip,
                n_bins,
                smote_method,
                smote_neighbors,
            ) = self.set_param_grid(trial)

            param_dict = None

        if use_weighted in ["sampler", "both"]:
            if not any([use_kmeans, use_kde]):
                raise optuna.exceptions.TrialPruned()

        if ModelClass in ["RF", "GB"]:
            trained_model, val_loss = self.run_rf_training(
                trial,
                param_dict,
                train_func,
                n_bins,
                smote_method,
                smote_neighbors,
                self.train_loader,
                self.val_loader,
            )
        else:
            criterion = WeightedRMSELoss()

            # Model, loss, and optimizer
            trained_model, val_loss = self.run_training(
                trial,
                criterion,
                ModelClass,
                train_func,
                self.dataset,
                lr,
                width,
                nlayers,
                dropout_prop,
                l2_weight,
                lr_scheduler_patience,
                lr_scheduler_factor,
                width_factor,
                grad_clip,
                n_bins,
                smote_method,
                smote_neighbors,
                self.train_loader,
                self.val_loader,
            )

            if trained_model is None:
                raise optuna.exceptions.TrialPruned()

        if np.isnan(val_loss):
            raise optuna.exceptions.TrialPruned()
        if val_loss is None:
            raise optuna.exceptions.TrialPruned()

        return val_loss

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

    def run_rf_training(
        self,
        trial,
        param_dict,
        train_func,
        n_bins,
        smote_method,
        smote_neighbors,
        train_loader,
        val_loader,
    ):
        return train_func(
            train_loader,
            val_loader,
            param_dict,
            trial,
            objective_mode=True,
            n_bins=n_bins,
            use_smote=self.args.use_synthetic_oversampling,
            smote_method=smote_method,
            smote_neighbors=smote_neighbors,
        )

    def run_training(
        self,
        trial,
        criterion,
        ModelClass,
        train_func,
        dataset,
        lr,
        width,
        nlayers,
        dropout_prop,
        l2_weight,
        lr_scheduler_patience,
        lr_scheduler_factor,
        width_factor,
        grad_clip,
        n_bins,
        smote_method,
        smote_neighbors,
        train_loader,
        val_loader,
    ):
        model = ModelClass(
            input_size=dataset.features.shape[1],
            width=width,
            nlayers=nlayers,
            dropout_prop=dropout_prop,
            device=self.device,
            factor=width_factor,
        ).to(self.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=l2_weight,
        )

        # Train model
        trained_model = train_func(
            train_loader,
            val_loader,
            model,
            criterion,
            optimizer,
            trial,
            lr_scheduler_factor,
            lr_scheduler_patience,
            objective_mode=True,
            grad_clip=grad_clip,
            use_smote=self.args.use_synthetic_oversampling,
            n_bins=n_bins,
            smote_neighbors=smote_neighbors,
            smote_method=smote_method,
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

    def set_gb_param_grid(self, trial, n_samples):
        gb_n_estimators = trial.suggest_int("gb_n_estimators", 50, 1000)
        gb_learning_rate = trial.suggest_float("gb_learning_rate", 0.01, 0.3)
        gb_subsample = trial.suggest_float("gb_subsample", 0.5, 1.0)
        gb_max_depth = trial.suggest_int("gb_max_depth", 3, 10)
        gb_min_child_weight = trial.suggest_int("gb_min_child_weight", 1, 10)
        gb_reg_alpha = trial.suggest_float("gb_reg_alpha", 0, 1.0)
        gb_reg_lambda = trial.suggest_float("gb_reg_lambda", 0.1, 1.0)
        gb_gamma = trial.suggest_float("gb_gamma", 5, 10)
        gb_colsample_bytree = trial.suggest_float("gb_colsample_bytree", 0.5, 1.0)

        n_bins = trial.suggest_int("n_bins", 3, 10)

        if self.args.force_no_weighting:
            l = ["none"]
        elif self.args.force_weighted_opt:
            l = [self.args.use_weighted]
        else:
            l = ["sampler", "none"]

        use_weighted = trial.suggest_categorical("use_weighted", l)

        if (
            self.args.use_synthetic_oversampling
            and self.args.oversample_method == "choose"
        ):
            l2 = ["kmeans", "optics", "kerneldensity"]
        elif self.args.use_synthetic_oversampling and self.args.oversample_method in [
            "kmeans",
            "optics",
            "kerneldensity",
        ]:
            l2 = [self.args.oversample_method]
        else:
            l2 = None

        smote_method = None
        smote_neighbors = self.args.oversample_neighbors
        if l2 is not None:
            smote_method = trial.suggest_categorical("oversample_method", l2)
            smote_neighbors = trial.suggest_int("oversample_neighbors", 2, 50)

        return (
            {
                "gb_n_estimators": gb_n_estimators,
                "gb_learning_rate": gb_learning_rate,
                "gb_subsample": gb_subsample,
                "gb_gamma": gb_gamma,
                "gb_max_depth": gb_max_depth,
                "gb_min_child_weight": gb_min_child_weight,
                "gb_reg_alpha": gb_reg_alpha,
                "gb_reg_lambda": gb_reg_lambda,
                "gb_colsample_bytree": gb_colsample_bytree,
            },
            n_bins,
            use_weighted,
            smote_method,
            smote_neighbors,
        )

    def set_rf_param_grid(self, trial, n_samples):
        n_estimators = trial.suggest_int("n_estimators", 50, 1000)
        criterion = trial.suggest_categorical(
            "criterion", ["squared_error", "absolute_error", "friedman_mse"]
        )
        max_depth = trial.suggest_int("max_depth", 3, n_samples)
        min_samples_split = trial.suggest_float("min_samples_split", 1e-3, 1.0)
        min_samples_leaf = trial.suggest_float("min_samples_leaf", 1e-3, 1.0)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        max_samples = trial.suggest_float("max_samples", 1e-3, 1.0)
        n_bins = trial.suggest_int("n_bins", 3, 5)

        if self.args.force_no_weighting:
            l = ["none"]
        elif self.args.force_weighted_opt:
            l = [self.args.use_weighted]
        else:
            l = ["sampler", "none"]

        use_weighted = trial.suggest_categorical("use_weighted", l)

        if (
            self.args.use_synthetic_oversampling
            and self.args.oversample_method == "choose"
        ):
            l2 = ["kmeans", "optics", "kerneldensity"]
        elif self.args.use_synthetic_oversampling and self.args.oversample_method in [
            "kmeans",
            "optics",
            "kerneldensity",
        ]:
            l2 = [self.args.oversample_method]
        else:
            l2 = None

        smote_method = None
        smote_neighbors = self.args.oversample_neighbors
        if l2 is not None:
            smote_method = trial.suggest_categorical("oversample_method", l2)
            smote_neighbors = trial.suggest_int("oversample_neighbors", 2, 50)

        return (
            {
                "n_estimators": n_estimators,
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
                "bootstrap": True,
                "oob_score": True,
                "max_samples": max_samples,
            },
            n_bins,
            use_weighted,
            smote_method,
            smote_neighbors,
        )

    def set_param_grid(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 0.5, log=True)
        width = trial.suggest_int(
            "width", 8, self.train_loader.dataset.tensors[0].shape[1] - 1
        )
        nlayers = trial.suggest_int("nlayers", 2, 20)
        dropout_prop = trial.suggest_float("dropout_prop", 0.0, 0.5)
        l2_weight = trial.suggest_float("l2_weight", 1e-6, 1e-1, log=True)

        lr_scheduler_patience = trial.suggest_int("lr_scheduler_patience", 10, 100)
        lr_scheduler_factor = trial.suggest_float("lr_scheduler_factor", 0.1, 1.0)
        width_factor = trial.suggest_float("factor", 0.2, 1.0)

        use_kde = True
        use_kmeans = False
        w_power = 1.0
        normalize = False

        # use_kmeans = trial.suggest_categorical("use_kmeans", [False, True])
        # use_kde = trial.suggest_categorical("use_kde", [False, True])
        # w_power = trial.suggest_int("w_power", 1, 10)

        # normalize = trial.suggest_categorical("normalize_sample_weights", [False, True])

        if self.args.force_no_weighting:
            l = ["none"]
        elif self.args.force_weighted_opt:
            l = [self.args.use_weighted]
        else:
            l = ["loss", "sampler", "both", "none"]

        use_weighted = trial.suggest_categorical("use_weighted", l)

        if (
            self.args.use_synthetic_oversampling
            and self.args.oversample_method == "choose"
        ):
            l2 = ["kmeans", "optics", "kerneldensity"]
        elif self.args.use_synthetic_oversampling and self.args.oversample_method in [
            "kmeans",
            "optics",
        ]:
            l2 = [self.args.oversample_method]
        else:
            l2 = None

        max_clusters = trial.suggest_int("max_clusters", 5, 100)
        max_neighbors = trial.suggest_int("max_neighbors", 5, 100)
        n_bins = trial.suggest_int("n_bins", 5, 20)

        smote_method = None
        smote_neighbors = self.args.oversample_neighbors
        if l2 is not None:
            smote_method = trial.suggest_categorical("oversample_method", l2)
            smote_neighbors = trial.suggest_int("oversample_neighbors", 2, 50)

        grad_clip = trial.suggest_categorical("grad_clip", [False, True])
        return (
            lr,
            width,
            nlayers,
            dropout_prop,
            l2_weight,
            lr_scheduler_patience,
            lr_scheduler_factor,
            width_factor,
            use_kmeans,
            use_kde,
            w_power,
            normalize,
            use_weighted,
            max_clusters,
            max_neighbors,
            grad_clip,
            n_bins,
            smote_method,
            smote_neighbors,
        )

    def stratified_weighted_multioutput_kfold(
        self, X, y, weights, n_splits=5, n_bins=10
    ):
        """
        Performs stratified K-Fold for multioutput regression data using inverse sample weights.

        Args:
            X (numpy.ndarray or pandas.DataFrame): Feature matrix.
            y (numpy.ndarray or pandas.DataFrame): Multioutput target matrix (e.g., longitude and latitude).
            weights (numpy.ndarray or pandas.Series): Inverse sample weights.
            n_splits (int): Number of folds. Default is 5.
            n_bins (int): Number of bins for stratification. Default is 10.

        Returns:
            List of tuples: Each tuple contains the train and test indices for each fold.
        """
        # Convert tensors to NumPy arrays for compatibility
        X_np = X.numpy()
        y_np = y.numpy()
        weights_np = weights.numpy()

        # Calculate composite metric, e.g., Euclidean distance from a central point (0,0)
        composite_metric = np.sqrt(y_np[:, 0] ** 2 + y_np[:, 1] ** 2)

        # Apply weights
        weighted_metric = composite_metric * weights_np

        # Bin the weighted composite metric
        y_binned = pd.qcut(weighted_metric, q=n_bins, labels=False, duplicates="drop")

        # Initialize StratifiedKFold
        skf = StratifiedKFold(n_splits=n_splits, random_state=self.args.seed)

        # Generate indices for each split
        fold_indices = [
            (train_idx, test_idx) for train_idx, test_idx in skf.split(X, y_binned)
        ]

        return fold_indices

    def evaluate_model(self, val_loader, model, criterion):
        model.eval()
        total_loss = 0.0
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
                total_loss += loss.item()
        return total_loss / len(val_loader)

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
            self.logger.info("Best trial parameters:", best_params)

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
