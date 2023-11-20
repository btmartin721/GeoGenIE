import logging

import numpy as np
from torch import from_numpy


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
