import json
import logging
import os
import pickle
from pathlib import Path

import torch
from optuna import create_study, pruners, samplers
from optuna.logging import (
    disable_default_handler,
    disable_propagation,
    enable_default_handler,
    enable_propagation,
)
from torch import optim

from geogenie.plotting.plotting import PlotGenIE


class Optimize:
    def __init__(
        self,
        train_loader,
        val_loader,
        test_loader,
        device,
        max_epochs,
        patience,
        prefix,
        output_dir,
        sqldb,
        n_trials,
        n_jobs,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=8,
        grad_clip=True,
        show_progress_bar=False,
        n_startup_trials=10,
        show_plots=False,
        fontsize=18,
        verbose=1,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.grad_clip = grad_clip
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

    def objective_function(self, trial, criterion, ModelClass, train_func):
        """Optuna hyperparameter tuning.

        Args:
            trial (optuna.Trial): Current Optuna trial.

        Returns:
            float: Loss value.
        """
        # Optuna hyperparameters
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        width = trial.suggest_int("width", 8, 512)
        nlayers = trial.suggest_int("nlayers", 1, 20)
        dropout_prop = trial.suggest_float("dropout_prop", 0.0, 0.5)
        l2_weight = trial.suggest_float("l2_weight", 1e-6, 1e-1, log=True)

        # Model, loss, and optimizer
        model = ModelClass(
            input_size=self.train_loader.dataset.tensors[0].shape[1],
            width=width,
            nlayers=nlayers,
            dropout_prop=dropout_prop,
            device=self.device,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_weight)

        # Train model
        trained_model = train_func(
            self.train_loader,
            self.val_loader,
            model,
            trial,
            criterion,
            optimizer,
            self.lr_scheduler_factor,
            self.lr_scheduler_patience,
            objective_mode=True,
            grad_clip=self.grad_clip,
        )

        # Evaluate model with test data.
        trained_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = trained_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(self.test_loader)

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
