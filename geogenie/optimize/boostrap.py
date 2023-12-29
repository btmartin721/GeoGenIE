import json
import logging
import os

import pandas as pd
import torch
from torch import optim

from geogenie.plotting.plotting import PlotGenIE


class Bootstrap:
    def __init__(
        self,
        train_loader,
        val_loader,
        val_indices,
        nboots,
        epochs,
        device,
        class_weights,
        width,
        nlayers,
        dropout_prop,
        lr,
        l2_weight,
        patience,
        output_dir,
        prefix,
        sample_data,
        verbose=1,
        show_plots=False,
        fontsize=18,
        filetype="png",
        dpi=300,
    ):
        """Class to run model with bootstrapping to estimate validation error.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            nboots (int): Number of bootstrap samples to create.
            epochs (int): Number of epochs for training each model.
            device (torch.device): Device to run the model on ('cpu' or 'cuda').
            class_weights (np.ndarray): Class weights for imbalanced sampling.
            width (int): Number of neurons in hidden layers.
            nlayers (int): Number of hidden layers.
            dropout_prop (float): Dropout proportion to reduce overfitting.
            lr (float): Learning rate for optimizer.
            l2_weight (float): L2 regularization weight (weight decay).
            patience (int): Patience to use for early stopping.
            prefix (str): Prefix for output filenames.
            sample_data (pd.DataFrame): Sample data with coordinates.
            verbose (int): Verbosity level.
            show_plots (bool): If True, shows in-line plots.
            fontsize (int): Fontsize for plot axis and title labels.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_indices = val_indices
        self.nboots = nboots
        self.epochs = epochs
        self.device = device
        self.class_weights = class_weights
        self.width = width
        self.nlayers = nlayers
        self.dropout_prop = dropout_prop
        self.lr = lr
        self.l2_weight = l2_weight
        self.patience = patience
        self.output_dir = output_dir
        self.prefix = prefix
        self.sample_data = sample_data
        self.verbose = verbose

        self.subdir = "bootstrap"
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

    def bootstrap_training_generator(self, ModelClass, train_func, criterion):
        """Generator for training models on bootstrapped samples.

        Args:
            ModelClass (torch.nn.Module): PyTorch model to train.
            train_func (callable): Callable function with PyTorch training loop.
            criterion (callable): PyTorch loss function to use.

        Yields:
            Trained model for each bootstrap sample.
            Training losses.
            Validation losses.
            Total train time.
        """
        for boot in range(self.nboots):
            try:
                self.logger.info(
                    f"Starting bootstrap iteration {boot + 1}/{self.nboots}"
                )

                # Resampling with replacement
                resampled_indices = torch.randint(
                    0, len(self.train_loader.dataset), (len(self.train_loader.dataset),)
                )

                # Obtain the weights corresponding to the resampled indices
                sampler = None
                if self.class_weights is not None:
                    resampled_weights = self.class_weights[resampled_indices]
                    sampler = torch.utils.data.WeightedRandomSampler(
                        resampled_weights, len(resampled_weights), replacement=True
                    )

                # Create a Subset of the dataset corresponding to the resampled indices
                resampled_dataset = torch.utils.data.Subset(
                    self.train_loader.dataset, resampled_indices
                )

                # Create a DataLoader for the resampled dataset with the new sampler
                resampled_loader = torch.utils.data.DataLoader(
                    resampled_dataset,
                    batch_size=self.train_loader.batch_size,
                    sampler=sampler,
                )

                # Reinitialize the model and optimizer each bootstrap
                model = ModelClass(
                    input_size=self.train_loader.dataset.tensors[0].shape[1],
                    width=self.width,
                    nlayers=self.nlayers,
                    dropout_prop=self.dropout_prop,
                    device=self.device,
                ).to(self.device)

                optimizer = optim.Adam(
                    model.parameters(), lr=self.lr, weight_decay=self.l2_weight
                )

                # Train the model.
                (
                    trained_model,
                    train_losses,
                    val_losses,
                    avgs,
                    stds,
                    total_train_time,
                ) = train_func(
                    resampled_loader,
                    self.val_loader,
                    model,
                    None,  # trial
                    criterion,
                    optimizer,
                    0.5,  # lr_scheduler_factor
                    self.patience // 6,  # lr_scheduler_patience
                    verbose=self.verbose,
                    grad_clip=True,
                )

                self.plotting.plot_times(
                    avgs,
                    stds,
                    os.path.join(
                        self.output_dir,
                        "plots",
                        f"{self.prefix}_bootrep{boot}.png",
                    ),
                )

                yield trained_model, total_train_time
            except Exception as e:
                self.logger.error(
                    f"Error during bootstrap iteration {boot + 1}: {e}",
                )
                raise

    def save_bootstrap_results(self, boot, val_metrics, val_preds, write_func):
        """
        Save the results of each bootstrap iteration.

        Args:
            boot (int): The current bootstrap iteration.
            val_metrics (dict): Validation metrics for the current iteration.
            val_preds (np.array): Predictions made by the model in the current iteration.
            write_func (callable): Function to write the predictions to file.

        Returns:
            pd.DataFrame: Output predictions.
        """
        outdir = os.path.join(self.output_dir, self.subdir)
        metrics_file_path = os.path.join(
            outdir,
            f"{self.prefix}_bootstrap_metrics.json",
        )

        boot_file_path = os.path.join(
            outdir,
            f"{self.prefix}_bootrep{boot}_metrics.txt",
        )

        with open(boot_file_path, "w") as fout:
            for k, v in val_metrics.items():
                fout.write(f"{k}: {v}\n")

        # If this is the first bootstrap iteration, delete existing file
        if boot == 0 and os.path.exists(metrics_file_path):
            if self.verbose >= 2:
                self.logger.warn("Found existing metrics file. Removing it.")
            os.remove(metrics_file_path)

        with open(metrics_file_path, "a") as fout:
            json.dump({boot: val_metrics}, fout, indent=2)
            fout.write("//")  # Add '//'' to separate JSON objects

        outfile = os.path.join(
            outdir,
            f"{self.prefix}_bootstrap_{boot}_predlocs.txt",
        )

        return write_func(
            val_preds,
            self.val_indices,
            self.sample_data,
            outfile,
        )

    def perform_bootstrap_training(
        self,
        train_func,
        pred_func,
        write_func,
        ModelClass,
        criterion,
        coord_scaler,
    ):
        """
        Perform training using bootstrap resampling.

        Args:
            train_func (callable): Callable PyTorch training loop.
            pred_func (callable): Callable predict function.
            write_func (callable): Callable function to write predictions to file.
            ModelClass (callable): Callble subclassed PyTorch model.
            coord_scaler (dict): Dictionary with meanlong, meanlat, sdlong, sdlat for scaling coordinates.

        Returns:
            None
        """
        try:
            if self.verbose >= 1:
                self.logger.info("Starting bootstrap training.")

            outdir = os.path.join(self.output_dir, "models")

            bootstrap_gen = self.bootstrap_training_generator(
                ModelClass, train_func, criterion
            )

            bootstrap_preds = []
            bootstrap_metrics = []
            train_times = []

            # Generator function. Does one bootstrap at a time.
            for boot, (model, train_time) in enumerate(bootstrap_gen):
                if self.verbose >= 1:
                    self.logger.info(
                        f"Processing bootstrap {boot + 1}/{self.nboots}",
                    )

                bootrep_file = os.path.join(
                    outdir, f"{self.prefix}_model_bootrep{boot}.pt"
                )

                # Save or evaluate the trained model
                torch.save(model.state_dict(), bootrep_file)

                # Evaluate model for each bootstrap replicate.
                val_preds, val_metrics = pred_func(
                    model,
                    self.val_loader,
                    device=self.device,
                )

                bootstrap_metrics.append(val_metrics)
                bootstrap_preds.append(val_preds)

                # Save metrics and predictions
                self.save_bootstrap_results(
                    boot,
                    val_metrics,
                    val_preds,
                    write_func,
                )

                train_times.append(train_time)

            outdir = os.path.join(self.output_dir, "plots")
            outfile = os.path.join(
                outdir,
                f"{self.prefix}_bootstrap_aggregation.png",
            )

            # Additional processing like plotting aggregate results, computing
            # statistics, etc.
            self.plotting.plot_bootstrap_aggregates(
                pd.DataFrame.from_dict(bootstrap_metrics), outfile
            )

            if self.verbose >= 1:
                self.logger.info("Bootstrap training completed.")

        except Exception as e:
            self.logger.error(
                f"Unexpected error in perform_bootstrap_training: {e}",
            )
            raise
