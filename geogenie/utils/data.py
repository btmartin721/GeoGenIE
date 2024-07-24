import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, features, labels=None, sample_weights=None, dtype=torch.float32):
        """
        Initialize custom PyTorch Dataset that incorporates sample weighting.

        Args:
            features (torch.Tensor): Input features.
            labels (torch.Tensor, optional): Labels corresponding to the features. Defaults to None.
            sample_weights (torch.Tensor): Weights for each sample. If None, then a sample_weights tensor is still created, but all weights will be equal to 1.0 (equal weighting). Defaults to None.
            dtype (torch.dtype): Data type to use with PyTorch. Must be a torch dtype. Defaults to torch.float32.

        Attributes:
            features (torch.Tensor): Input features.
            labels (torch.Tensor): Labels corresponding to features.
            sample_weights (torch.Tensor): Sample weights of shape (n_samples,).
            tensors (tuple): Tuple consisting of (features, labels, sample_weights).
        """
        self.dtype = dtype

        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

        self.tensors = (self.features, self.labels, self.sample_weights)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._features = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._labels = value

    @property
    def sample_weights(self):
        return self._sample_weights

    @sample_weights.setter
    def sample_weights(self, value):
        if value is None:
            value = torch.ones(len(self.features), dtype=self.dtype)
        elif not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype)
        self._sample_weights = value

    @property
    def n_features(self):
        """
        Return the number of columns in the features dataset.
        """
        return self.features.shape[1] if self.features.ndimension() > 1 else 1

    @property
    def n_labels(self):
        """
        Return the number of columns in the labels dataset.
        """
        return self.labels.shape[1] if self.labels.ndimension() > 1 else 1

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Retrieve the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (feature, label, sample_weight) for the specified index.
        """
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx], self.sample_weights[idx]
