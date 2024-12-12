import logging
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """Early stopping PyTorch callback.

    This class defines an early stopping callback for PyTorch models.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (int): Verbosity mode.
        delta (float): Minimum change to qualify as an improvement.
        output_dir (str): Directory to save the outputs.
        prefix (str): Prefix for the saved files.
        counter (int): Counter for the number of epochs since last improvement.
        early_stop (bool): Flag to indicate if early stopping is triggered.
        best_score (float): Best score for the monitored quantity.
        val_loss_min (float): Minimum validation loss.
        boot (int): Boot object or identifier.
        trial (int): Trial number for hyperparameter optimization.
        logger (logging.Logger): Logger object for the class.
    """

    def __init__(
        self,
        output_dir,
        prefix,
        patience=100,
        verbose=0,
        delta=0,
        trial=None,
        boot=None,
    ):
        """Initialize the EarlyStopping callback.

        Args:
            output_dir (str): Directory to save checkpoints.
            prefix (str): Prefix for the checkpoint filenames.
            patience (int): How long to wait after last time validation loss improved. Default: 100.
            verbose (int): Verbosity mode. Default: 0.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0.
            trial (optuna.trial.Trial, optional): Optuna trial for hyperparameter optimization. Default: None.
            boot (int, optional): Bootstrap number for ensemble models. Default: None.

        Raises:
            ValueError: If both boot and trial are defined.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.output_dir = output_dir
        self.prefix = prefix

        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.val_loss_min = np.Inf

        self.boot = boot
        self.trial = trial.number if trial is not None else None

        self.logger = logging.getLogger(__name__)

        if self.boot is not None and self.trial is not None:
            msg = "Both boot and trial cannot both be defined."
            self.logger.error(msg)
            raise ValueError(msg)

    def __call__(self, val_loss, model):
        """Call method to check if early stopping condition is met.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases.

        Args:
            val_loss (float): Current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        if self.verbose:
            self.logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        chkdir = Path(self.output_dir, "models")
        chkdir.mkdir(parents=True, exist_ok=True)

        if self.boot is not None:
            if self.verbose:
                self.logger.info(f"Saving checkpoint for boot {self.boot}")
            fn = chkdir / f"{self.prefix}_boot{self.boot}_checkpoint.pt"
        else:
            if self.verbose:
                self.logger.info(f"Saving checkpoint for trial {self.trial}")
            fn = chkdir / f"{self.prefix}_trial{self.trial}_checkpoint.pt"

        torch.save(model.state_dict(), fn)
        self.val_loss_min = val_loss

    def load_best_model(self, model):
        """Load the best model from the checkpoint file.

        Args:
            model (torch.nn.Module): The model to load the checkpoint into.

        Returns:
            torch.nn.Module: The model with weights loaded from the best checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
        """
        chkdir = Path(self.output_dir, "models")

        if self.boot is not None:
            fn = chkdir / f"{self.prefix}_boot{self.boot}_checkpoint.pt"
        else:
            fn = chkdir / f"{self.prefix}_trial{self.trial}_checkpoint.pt"

        if fn.exists():
            model.load_state_dict(torch.load(fn))
            if self.verbose:
                self.logger.info("Loaded the best model from checkpoint.")
            return model
        else:
            msg = f"Checkpoint file {fn} not found. Early stopping failed and model not loaded."
            self.logger.error(msg)
            raise FileNotFoundError(msg)


def callback_init(optimizer, args, trial=None, boot=None):
    """Initialize early stopping and learning rate scheduler callbacks.

        EarlyStopping Arguments:
        output_dir (str): Directory to save the outputs.
        prefix (str): Prefix for the saved files.
        patience (int): Number of epochs to wait for improvement before stopping.
        verbose (bool): If True, prints messages about early stopping.
        delta (float): Minimum change to qualify as an improvement.
        trial (optuna.trial.Trial): Optuna trial object.
        boot (any): Boot object or identifier.

    ReduceLROnPlateau Arguments:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        mode (str): One of 'min', 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        verbose (bool): If True, prints a message to stdout for each update.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the learning rate scheduler will be applied.
        args (argparse.Namespace): Argument namespace containing the necessary hyperparameters and settings.
        trial (optuna.trial.Trial, optional): Optuna trial object for hyperparameter optimization. Defaults to None.
        boot (any, optional): Boot object or identifier, used for early stopping callback. Defaults to None.

    Returns:
        tuple: A tuple containing the initialized EarlyStopping and ReduceLROnPlateau scheduler.
    """

    verbose = args.verbose >= 2 or args.debug

    early_stopping = EarlyStopping(
        output_dir=args.output_dir,
        prefix=args.prefix,
        patience=args.early_stop_patience,
        verbose=verbose,
        delta=0,
        trial=trial,
        boot=boot,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        verbose=verbose,
    )

    return early_stopping, lr_scheduler
