import logging
import os

import numpy as np
from torch import save as torchsave
from torch.optim.lr_scheduler import ReduceLROnPlateau


class EarlyStopping:
    """Early stopping pytorch callback."""

    def __init__(
        self, output_dir, prefix, patience=100, verbose=0, delta=0, trial=None
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.output_dir = output_dir
        self.prefix = prefix
        self.trial = 0 if trial is None else trial.number

        self.logger = logging.getLogger(__name__)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose >= 2:
                self.logger.info(
                    f"EarlyStopping counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose >= 2:
            self.logger.info(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        checkpoint_fn = os.path.join(
            self.output_dir, "models", f"{self.prefix}_trial{self.trial}_checkpoint.pt"
        )
        torchsave(model.state_dict(), checkpoint_fn)
        self.val_loss_min = val_loss


def callback_init(optimizer, args, trial=None):
    early_stopping = EarlyStopping(
        output_dir=args.output_dir,
        prefix=args.prefix,
        patience=args.early_stop_patience,
        verbose=args.verbose >= 2,
        delta=0,
        trial=trial,
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        verbose=args.verbose >= 2,
    )

    return early_stopping, lr_scheduler
