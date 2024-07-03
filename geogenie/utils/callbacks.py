import logging
from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """Early stopping PyTorch callback."""

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
        """
        Args:
            output_dir (str): Directory to save checkpoints.
            prefix (str): Prefix for the checkpoint filenames.
            patience (int): How long to wait after last time validation loss improved. Default: 100
            verbose (int): Verbosity mode. Default: 0
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            trial (optuna.trial): Optuna trial for hyperparameter optimization. Default: None
            boot (int): Bootstrap number for ensemble models. Default: None
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
        """Saves model when validation loss decreases."""
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
        """Loads the best model from the checkpoint file."""
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
    early_stopping = EarlyStopping(
        output_dir=args.output_dir,
        prefix=args.prefix,
        patience=args.early_stop_patience,
        verbose=args.verbose >= 2,
        delta=0,
        trial=trial,
        boot=boot,
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        verbose=args.verbose >= 2,
    )

    return early_stopping, lr_scheduler
