import torch


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, sample_weights=None, dtype=torch.float32):
        """
        Initialize custom PyTorch Dataset that incorporates sample weighting.

        Args:
            features (Tensor): Input features.
            labels (Tensor): Labels corresponding to the features.
            sample_weights (Tensor): Weights for each sample. If None, then a sample_weights tensor is still created, but all weights will be equal to 1.0 (equal weighting). Defaults to None.
            dtype (torch.dtype): Data type to use with PyTorch. Must be a torch dtype. Defaults to torch.float32.

        Attributes:
            features (torch.Tensor): Input features.
            labels (torch.Tensor): Labels corresponding to features.
            sample_weights (torch.Tensor): Sample weights of shape (n_samples,).
            tensors (tuple): Tuple consisting of (features, labels, sample_weights).
        """
        self.dtype = dtype

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=self.dtype)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=self.dtype)

        if sample_weights is None:
            sample_weights = torch.ones(len(features), dtype=self.dtype)

        if not isinstance(sample_weights, torch.Tensor):
            sample_weights = torch.tensor(sample_weights, dtype=self.dtype)

        self.features = features
        self.labels = labels
        self.sample_weights = sample_weights

        # Store the tensors as a tuple
        self.tensors = (features, labels, sample_weights)

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

        return self.features[idx], self.labels[idx], self.sample_weights[idx]
