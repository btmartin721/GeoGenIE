import logging

import numpy as np
from sklearn.neighbors import KernelDensity
from torch import from_numpy
from torch.utils.data import Sampler


class Samplers:
    def __init__(self, train_indices, verbose=1):
        self.train_indices = train_indices
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

    def get_class_weights_populations(self, popmap_data):
        """Get class weights for torch.utils.data.WeightedRandomSampler."""

        if self.verbose >= 1:
            self.logger.info("Using class weighting by population IDs.")

        weight_train = popmap_data["populationID"].iloc[self.train_indices].to_numpy()
        unique_pops = np.unique(weight_train)
        pop_to_index = {pop: idx for idx, pop in enumerate(unique_pops)}

        class_sample_count = np.array(
            [len(np.where(weight_train == pop)[0]) for pop in unique_pops]
        )

        weight = 1.0 / class_sample_count
        samples_weight = np.array(
            [weight[pop_to_index[pop]] for pop in weight_train],
        )

        samples_weight = from_numpy(samples_weight)  # pytorch.
        return samples_weight


class GeographicDensitySampler(Sampler):
    """
    A PyTorch sampler that samples based on geographic density.

    Attributes:
        indices (list of int): List of indices to sample from.
        weights (numpy array): Array of sample weights.
    """

    def __init__(self, data, bandwidth=0.1):
        """
        Args:
            data (pandas DataFrame): DataFrame containing 'x' (longitude) and 'y' (latitude).
            bandwidth (float): Bandwidth for KDE. Adjust for more/less smoothing.
        """
        self.data = data
        self.bandwidth = bandwidth
        self.indices = np.arange(len(data))
        self.weights = self.calculate_weights()

    def calculate_weights(self):
        """
        Calculate weights based on geographic density using Kernel Density Estimation.

        Returns:
            numpy array: Weights for each sample.
        """
        coords = self.data.copy()  # Columns ['x' and 'y']
        kde = KernelDensity(bandwidth=self.bandwidth, kernel="gaussian")
        kde.fit(coords)
        log_density = kde.score_samples(coords)
        density = np.exp(log_density)

        # More aggressive transformation: square the inverse density
        weights = 1 / np.square(density + 1e-5)
        return weights

    def __iter__(self):
        return (
            self.indices[i]
            for i in np.random.choice(
                self.indices,
                size=len(self.indices),
                p=self.weights / np.sum(self.weights),
            )
        )

    def __len__(self):
        return len(self.data)
