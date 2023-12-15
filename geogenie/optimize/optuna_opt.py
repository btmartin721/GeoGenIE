import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from optuna import create_study, pruners, samplers
from optuna.logging import (
    disable_default_handler,
    enable_default_handler,
    enable_propagation,
)
from sklearn.model_selection import KFold
from torch import optim
from torch.utils.data import DataLoader, Subset

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import GeographicDensitySampler
from geogenie.utils.loss import MultiobjectiveHaversineLoss
from geogenie.utils.utils import CustomDataset


class Optimize:
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
        show_progress_bar=False,
        n_startup_trials=10,
        show_plots=False,
        fontsize=18,
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
        self.show_progress_bar = show_progress_bar
        self.n_startup_trials = n_startup_trials
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

        self.plotting = PlotGenIE(
            device, output_dir, prefix, show_plots=show_plots, fontsize=fontsize
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
        dataset = self.train_loader.dataset

        # Optuna hyperparameters
        lr = trial.suggest_float("lr", 0.0, 0.5)
        width = trial.suggest_int(
            "width", 2, self.train_loader.dataset.tensors[0].shape[1] - 1
        )
        nlayers = trial.suggest_int("nlayers", 2, 20)
        dropout_prop = trial.suggest_float("dropout_prop", 0.0, 0.5)
        l2_weight = trial.suggest_float("l2_weight", 1e-6, 1e-1, log=True)
        alpha = trial.suggest_float("alpha", 0.2, 1.0)
        beta = trial.suggest_float("beta", 0.2, 1.0)
        gamma = trial.suggest_float("gamma", 0.2, 1.0)
        lr_scheduler_patience = trial.suggest_int("lr_scheduler_patience", 10, 100)
        lr_scheduler_factor = trial.suggest_float("lr_scheduler_factor", 0.1, 1.0)
        width_factor = trial.suggest_float("factor", 0.2, 1.0)
        use_kmeans = trial.suggest_categorical("use_kmeans", [False, True])
        use_kde = trial.suggest_categorical("use_kde", [False, True])
        w_power = trial.suggest_int("w_power", 1, 10)
        use_weighted = trial.suggest_categorical(
            "use_weighted", ["loss", "sampler", "both", "none"]
        )
        max_clusters = trial.suggest_int("max_clusters", 5, 100)
        max_neighbors = trial.suggest_int("max_neighbors", 5, 100)

        grad_clip = trial.suggest_categorical("grad_clip", [False, True])

        # K-Fold Cross-Validation
        num_folds = 5
        kfold = KFold(n_splits=num_folds, shuffle=False)
        fold_losses = []  # Store losses for each fold
        gwrs = []
        total_loss = 0.0

        kwargs = {"batch_size": self.train_loader.batch_size}

        for train_idx, val_idx in kfold.split(dataset):
            # Create train and validation subsets for this fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            # Initialize lists to store features and labels
            subset_features = []
            subset_labels = []

            if use_weighted in ["loss", "both"]:
                subset_weights = []
            else:
                subset_weights = torch.ones(len(train_subset), dtype=torch.float32)
            # Iterate over the subset and extract features and labels
            for data in train_subset:
                features, labels, sample_weights = data
                subset_features.append(features)
                subset_labels.append(labels)

                if use_weighted in ["loss", "both"]:
                    subset_weights.append(sample_weights)

            subset_features = torch.stack(subset_features).to(torch.float32)
            subset_labels = torch.stack(subset_labels).to(torch.float32)

            if not isinstance(subset_weights, torch.Tensor):
                subset_weights = torch.stack(subset_weights).to(torch.float32)

            # Get subset of weighted sampler.
            weighted_sampler = GeographicDensitySampler(
                pd.DataFrame(subset_labels.numpy(), columns=["x", "y"]),
                use_kmeans=use_kmeans,
                use_kde=use_kde,
                w_power=w_power,
                max_clusters=max_clusters,
                max_neighbors=max_neighbors,
                objective_mode=True,
            )

            train_subset_dataset = CustomDataset(
                subset_features, subset_labels, weighted_sampler.weights
            )

            if use_weighted in ["sampler", "both"]:
                kwargs["sampler"] = weighted_sampler
            else:
                kwargs["shuffle"] = True
            train_loader = DataLoader(train_subset_dataset, **kwargs)

            val_loader = DataLoader(
                val_subset, batch_size=kwargs["batch_size"], shuffle=False
            )

            # Model, loss, and optimizer
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

            criterion = MultiobjectiveHaversineLoss(alpha, beta, gamma)

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
            )

            # Evaluate on validation set
            fold_loss, gwr = self.evaluate_model(
                val_loader,
                trained_model,
                criterion,
                weighted_sampler,
            )
            fold_losses.append(fold_loss)
            gwrs.append(gwr)
            total_loss += fold_loss

        # Calculate average loss and standard deviation
        average_loss = np.mean(fold_losses)
        std_dev = np.std(fold_losses)

        average_gwr = np.mean(gwrs)
        std_dev_gwr = np.std(gwrs)

        # Create a new row as a DataFrame
        new_row = pd.DataFrame(
            [
                {
                    "trial": trial.number,
                    "average_loss": average_loss,
                    "std_dev": std_dev,
                    "average_gwr": average_gwr,
                    "std_dev_gwr": std_dev_gwr,
                },
            ]
        )

        # Concatenate the new row to the existing DataFrame
        self.cv_results = pd.concat(
            [self.cv_results, new_row],
            ignore_index=True,
        )

        return average_loss

    def evaluate_model(self, val_loader, model, criterion, weighted_sampler):
        model.eval()
        total_loss = 0.0
        total_gwr = 0.0
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
                loss = criterion(outputs, labels, sample_weight=sample_weights)
                gwr = weighted_sampler.perform_gwr(
                    outputs.numpy(), labels.numpy(), sample_weights.numpy()
                )
                total_loss += loss.item()
                total_gwr += gwr
        return total_loss / len(val_loader), gwr / len(val_loader)

    def perform_optuna_optimization(self, criterion, ModelClass, train_func):
        """
        Perform parameter optimization using Optuna.

        Args:
            logger (logging): Logger to write output to.

        Returns:
            tuple: Best trial and the Optuna study object.
        """
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
            n_ei_candidates=self.patience // 2,
        )
        pruner = pruners.MedianPruner()
        Path(self.sqldb).mkdir(parents=True, exist_ok=True)

        storage_path = f"sqlite:///{self.sqldb}/{self.prefix}_optuna.db"
        if self.verbose >= 1:
            self.logger.info(f"Writing Optuna data to database: {storage_path}")

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

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=self.show_progress_bar,
        )

        cv_outfile = os.path.join(
            self.output_dir, "optimize", f"{self.prefix}_cv_results.csv"
        )

        self.cv_results.to_csv(cv_outfile, header=True, index=False)

        if self.verbose >= 1:
            self.logger.info("Finished parameter search!")

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
