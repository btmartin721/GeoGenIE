import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """Class to create a custom PyTorch Dataset with sample weighting and sample IDs.

    This class defines a custom PyTorch Dataset that incorporates sample weighting and sample IDs.

    Attributes:
        tensors (tuple): Tuple consisting of (features, labels, sample_weights, sample_ids).
    """

    def __init__(
        self,
        features,
        labels=None,
        sample_weights=None,
        sample_ids=None,
        dtype=torch.float32,
    ):
        """Initialize custom PyTorch Dataset that incorporates sample weighting and sample IDs.

        Args:
            features (torch.Tensor): Input features.
            labels (torch.Tensor, optional): Labels corresponding to the features. Defaults to None.
            sample_weights (torch.Tensor, optional): Weights for each sample. If None, then a sample_weights tensor is still created, but all weights will be equal to 1.0 (equal weighting). Defaults to None.
            sample_ids (list or array-like, optional): Unique identifiers for each sample. If None, indices will be used as sample IDs. Defaults to None.
            dtype (torch.dtype): Data type to use with PyTorch. Must be a torch dtype. Defaults to torch.float32.
        """
        self.dtype = dtype

        self._features = features
        self._labels = labels
        self._sample_weights = sample_weights

        if self._sample_weights is None:
            self._sample_weights = torch.ones(len(self._features), dtype=self.dtype)

        self._sample_ids = sample_ids

        self.tensors = (self.features, self.labels, self.sample_weights)

    @property
    def features(self):
        """Get the features tensor."""
        if not isinstance(self._features, torch.Tensor):
            self._features = torch.tensor(self._features, dtype=self.dtype)
        return self._features

    @features.setter
    def features(self, value):
        """Set the features tensor."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._features = value

    @property
    def labels(self):
        """Get the labels tensor."""
        if not isinstance(self._labels, torch.Tensor) and self._labels is not None:
            self._labels = torch.tensor(self._labels, dtype=self.dtype)
        return self._labels

    @labels.setter
    def labels(self, value):
        """Set the labels tensor."""
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._labels = value

    @property
    def sample_weights(self):
        """Get the sample weights."""
        if (
            not isinstance(self._sample_weights, torch.Tensor)
            and self._sample_weights is not None
        ):
            self._sample_weights = torch.tensor(self._sample_weights, dtype=self.dtype)

        if self._sample_weights is None:
            self._sample_weights = torch.ones(len(self.features), dtype=self.dtype)
        return self._sample_weights

    @sample_weights.setter
    def sample_weights(self, value):
        """Set the sample weights."""
        if value is None:
            value = torch.ones(len(self.features), dtype=self.dtype)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._sample_weights = value

    @property
    def sample_ids(self):
        """Get the sample IDs."""
        return self._sample_ids

    @sample_ids.setter
    def sample_ids(self, value):
        """Set the sample IDs."""
        if value is None:
            value = list(range(len(self.features)))  # Use indices as default sample IDs
        self._sample_ids = value

    @property
    def n_features(self):
        """Return the number of columns in the features dataset."""
        return self.features.shape[1] if self.features.ndimension() > 1 else 1

    @property
    def n_labels(self):
        """Return the number of columns in the labels dataset."""
        return self.labels.shape[1] if self.labels.ndimension() > 1 else 1

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """Retrieve the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (feature, label, sample_weight, sample_id) for the specified index.
        """
        if self.sample_ids is not None:
            if self.labels is None:
                return self.features[idx], self.sample_ids[idx]
            return (
                self.features[idx],
                self.labels[idx],
                self.sample_weights[idx],
                self.sample_ids[idx],
            )
        else:
            sample_ids = list(range(len(self.features)))

            if self.labels is None:
                return self.features[idx], sample_ids[idx]
            return (
                self.features[idx],
                self.labels[idx],
                self.sample_weights[idx],
                sample_ids[idx],
            )
