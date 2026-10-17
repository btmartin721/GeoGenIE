import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
from geogenie.samplers.samplers import (
    GeographicDensitySampler,
    as_generator,
    sklearn_seed,
)
from geogenie.utils.data import CustomDataset


class GenotypeInterpolator:
    """Class to interpolate genotypes and geographic coordinates among nearest neighbors.

    This class interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

    Attributes:
        X (np.ndarray): 2D array of genotypes.
        y (np.ndarray): 2D array of normalized geographic coordinates.
        sample_weights (np.ndarray): 1D array of sample weights.
        args (argparse.Namespace): User-supplied (or default) arguments.
        embedded (bool): Indicates if the genotypes are embedded (dimensionality reduced).
        use_kde_densities (bool): Flag to use KernelDensity for density estimation in clustering.
        n_bins (int or None): Number of clusters to use with KMeans clustsering.
        verbose (int): Verbosity setting (0-3).
        fontsize (int): Font size for plots.
        dpi (int): Dots per inch for plots.
        plot_type (str): File type for plots.
        logger (logging.Logger): Logger object.
        n_neighbors (int): Number of neighbors to consider for each sample.
        kde (KernelDensity): KernelDensity estimator for density estimation.
        bandwidth (float): Optimal bandwidth for KernelDensity.
        threshold (float): Optimal threshold for KernelDensity.
        cluster_labels (np.ndarray): Cluster labels for each sample.
        original_cluster_labels (np.ndarray): Original cluster labels for each sample.
        neighbor_indices (np.ndarray): Indices of the nearest neighbors for each sample (the sample itself is excluded).
        max_oversample_factor (int or None): Upper limit on the rows (original plus synthetic) produced per original sample.
        rng (np.random.Generator): Random number generator used for all random draws.
        oversample_factors (dict): Oversampling factor per cluster, set by ``interpolate_genotypes``.
        sample_origin_list (np.ndarray): "original" or "synthetic" for each output row, set by ``interpolate_genotypes``.
        parent_indices (np.ndarray): Index (into ``X``) of the parent of each output row; for original rows, the row itself.
        mate_indices (np.ndarray): Index (into ``X``) of the second parent of each synthetic row; -1 for original rows.
        plotting (PlotGenIE): PlotGenIE object for visualization
    """

    def __init__(
        self,
        X,
        y,
        sample_weights,
        args,
        n_neighbors=None,
        embedded=False,
        use_kde_densities=False,
        n_bins=None,
        verbose=0,
        fontsize=24,
        dpi=300,
        plot_type="png",
        max_oversample_factor=5,
        seed=None,
    ):
        """Initialize the GenotypeInterpolator with genotype data and geographic coordinates.

        This class interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

        Args:
            X (np.ndarray): 2D array of genotypes.
            y (np.ndarray): 2D array of geographic coordinates (longitude, latitude) in degrees.
            sample_weights (np.ndarray): 1D array of sample weights.
            args (argparse.Namespace): User-supplied (or default) arguments.
            n_neighbors (int or str, optional): Number of neighbors (excluding the sample itself) to consider for each sample: an integer >= 1, 'sqrt', or 'log2'. If None, uses 'sqrt' (square root of the number of samples). Defaults to None.
            embedded (bool): Not supported; ignored with a warning. Continuous (e.g., embedded) features are interpolated as the midpoint of the parents. Defaults to False.
            use_kde_densities (bool): Flag to use KernelDensity for density estimation in clustering. Defaults to False.
            n_bins (int or None): Number of clusters to use with KMeans clustsering. If None, then the number of bins are estimated from the data. Not used when ``use_kde_densities`` is True (two groups: sparse and dense). Defaults to None.
            verbose (int): Verbosity setting (0-3). Defaults to 0 (silent).
            max_oversample_factor (int or None, optional): Maximum oversampling factor per cluster, i.e. at most ``max_oversample_factor - 1`` synthetic samples per original sample. None means no limit. Defaults to 5.
            seed (int, np.random.Generator, or None, optional): Seed or Generator for all random draws (KMeans, neighbor choice, Mendelian sampling, shuffling). If None, a fresh, unseeded Generator is used. Defaults to None.
        """
        self.X = X
        self.y = y
        self.sample_weights = sample_weights
        self.args = args
        self.embedded = embedded
        self.use_kde_densities = use_kde_densities
        self.n_bins = n_bins
        self.verbose = verbose
        self.fontsize = fontsize
        self.dpi = dpi
        self.plot_type = plot_type
        self.rng = as_generator(seed)
        self._warned_continuous = False

        self.logger = logging.getLogger(__name__)

        if max_oversample_factor is not None:
            if (
                isinstance(max_oversample_factor, bool)
                or not isinstance(max_oversample_factor, (int, np.integer))
                or max_oversample_factor < 1
            ):
                msg = f"'max_oversample_factor' must be None or an integer >= 1, but got: {max_oversample_factor}"
                self.logger.error(msg)
                raise ValueError(msg)
            max_oversample_factor = int(max_oversample_factor)
        self.max_oversample_factor = max_oversample_factor

        if embedded:
            self.logger.warning(
                "'embedded=True' is not supported and is ignored. Continuous features are interpolated as the midpoint of the two parents."
            )
            self.embedded = False

        if n_neighbors is not None:
            is_int = isinstance(n_neighbors, (int, np.integer)) and not isinstance(
                n_neighbors, bool
            )
            if not is_int and (
                not isinstance(n_neighbors, str)
                or n_neighbors.lower() not in {"sqrt", "log2"}
            ):
                msg = f"'n_neighbors' must be either 'sqrt', 'log2', NoneType, or an integer, but got: {n_neighbors}."
                self.logger.error(msg)
                raise ValueError(msg)
            elif isinstance(n_neighbors, str):
                n_neighbors = n_neighbors.lower()
            else:
                if n_neighbors < 1:
                    msg = f"'n_neighbors must be >= 1, but got: {n_neighbors}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                n_neighbors = int(n_neighbors)

        else:
            n_neighbors = "sqrt"

        self.kde = None
        if use_kde_densities:
            # self.kde gets defined in _automated_parameter_tuning.
            self.bandwidth, self.threshold = self._automated_parameter_tuning()
            if self.kde is None:
                msg = "KernelDensity estimator did not get fitted."
                self.logger.error(msg)
                raise AttributeError(msg)
            if self.threshold is None:
                msg = "No KernelDensity threshold split the samples into a sparse and a dense group."
                self.logger.error(msg)
                raise ValueError(msg)
            if n_bins is not None and self.verbose >= 1:
                self.logger.info(
                    "'n_bins' is not used with KernelDensity oversampling (two groups: sparse and dense)."
                )

        self.n_neighbors = self._determine_optimal_neighbors(n_neighbors)

        if self.verbose >= 1:
            if use_kde_densities:
                self.logger.info(
                    f"Optiaml bandwidth for KernelDensity: {self.bandwidth}"
                )
                self.logger.info(
                    f"Optiaml threshold for KernelDensity: {self.threshold}"
                )

        self.use_kde_densities = use_kde_densities

        if use_kde_densities:
            self.cluster_labels = self._perform_density_estimation()
            self.centroids = None
        else:
            self.cluster_labels, self.centroids = self._perform_kmeans_clustering()

        self.original_cluster_labels = self.cluster_labels.copy()
        self.neighbor_indices = self._find_nearest_neighbors()

        self.plotting = PlotGenIE(
            "cpu",
            args.output_dir,
            args.prefix,
            args.basemap_fips,
            args.highlight_basemap_counties,
            args.shapefile,
            show_plots=args.show_plots,
            fontsize=args.fontsize,
            filetype=args.filetype,
            dpi=args.plot_dpi,
            remove_splines=args.remove_splines,
        )

    def _determine_optimal_neighbors(self, n_neighbors):
        """Determine an optimal number of neighbors based on the dataset.

        Args:
            n_neighbors (int or str): Number of neighbors to consider for each sample: an integer, 'sqrt', or 'log2'.

        Returns:
            int: Estimated optimal number of neighbors, between 1 and n_samples - 1.

        Raises:
            TypeError: If an invalid type is provided for n_neighbors.
            ValueError: If there are fewer than two samples.
        """
        n_samples = len(self.y)
        if n_samples < 2:
            msg = f"At least two samples are required for genotype interpolation, but got: {n_samples}"
            self.logger.error(msg)
            raise ValueError(msg)

        if isinstance(n_neighbors, str):
            method = np.sqrt if n_neighbors == "sqrt" else np.log2
            # Simple heuristic: square root of the number of samples
            optimal_n = int(method(n_samples))
        elif isinstance(n_neighbors, (int, np.integer)):
            optimal_n = int(n_neighbors)
        else:
            msg = f"Expected str or int, but got: {type(n_neighbors)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if optimal_n > n_samples - 1:
            self.logger.warning(
                f"'n_neighbors' ({optimal_n}) is larger than the number of other samples; using {n_samples - 1}."
            )
        return int(np.clip(optimal_n, 1, n_samples - 1))

    def _find_nearest_neighbors(self):
        """Find the nearest neighbors for each sample based on geographic coordinates.

        The sample itself is excluded from its own neighbor list, so every synthetic sample has two different parents.

        Returns:
            np.ndarray: Indices of the nearest neighbors for each sample, shape (n_samples, n_neighbors).
        """
        n_samples = len(self.y)
        nbrs = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, n_samples), algorithm="auto"
        )
        nbrs.fit(self.y)
        _, indices = nbrs.kneighbors(self.y)

        # Drop the sample itself. If it is not in its own list (more than
        # n_neighbors samples share its coordinates), drop the last column.
        is_self = indices == np.arange(n_samples)[:, None]
        is_self[~is_self.any(axis=1), -1] = True
        indices = indices[~is_self].reshape(n_samples, -1)

        return indices

    def interpolate_genotypes(self):
        """Perform synthetic over-sampling of the data by interpolating genotypes and geographic coordinates among nearest neighbors.

        Each cluster gets an oversampling factor ``largest_cluster_size // cluster_size``, capped at ``max_oversample_factor``. Each original sample is kept and ``factor - 1`` synthetic samples are added; each synthetic sample has the original sample and a neighbor (drawn separately for each synthetic sample, without replacement where possible, never the sample itself) as parents. Its coordinates are the parents' midpoint and its sample weight is the mean of the parents' weights.

        Returns:
            np.ndarray: Synthetically over-sampled genotype array and corresponding geographic coordinates.
        """

        # Identify the number of samples in each cluster
        # Use the labels of the original samples, so repeated calls give the
        # same oversampling scheme.
        original_labels = self.original_cluster_labels
        unique_clusters, counts = np.unique(original_labels, return_counts=True)
        largest_cluster_size = counts.max()

        # Calculate the oversampling factor for each cluster
        oversample_factors = {}
        for cluster, count in zip(unique_clusters, counts):
            factor = int(largest_cluster_size // count)
            if (
                self.max_oversample_factor is not None
                and factor > self.max_oversample_factor
            ):
                self.logger.info(
                    f"Oversampling factor for cluster {cluster} (n={count}) capped from {factor} to {self.max_oversample_factor}."
                )
                factor = self.max_oversample_factor
            oversample_factors[cluster] = factor
        self.oversample_factors = oversample_factors

        # Lists to store over-sampled data and metadata
        over_sampled_X_list = []
        over_sampled_y_list = []
        over_sampled_S_list = []
        parent_list = []
        mate_list = []
        sample_origin_list = []  # Keep track of original and synthetic

        for i, genotype in enumerate(self.X):
            cluster = original_labels[i]
            num_samples_to_generate = oversample_factors.get(
                cluster, 1
            )  # Get the oversampling factor for the current cluster

            # Add the original sample to the lists
            over_sampled_X_list.append(genotype)
            over_sampled_y_list.append(self.y[i])
            over_sampled_S_list.append(self.sample_weights[i])
            parent_list.append(i)
            mate_list.append(-1)
            sample_origin_list.append("original")  # Mark as original

            # Subtract 1 to account for the original sample
            n_synthetic = num_samples_to_generate - 1
            if n_synthetic < 1:
                continue

            # Choose a neighbor for each synthetic sample
            candidates = self.neighbor_indices[i]
            neighbor_choices = self.rng.choice(
                candidates, size=n_synthetic, replace=n_synthetic > len(candidates)
            )

            # Generate the synthetic samples
            for neighbor_index in neighbor_choices:
                neighbor_genotype = self.X[neighbor_index]
                neighbor_coords = self.y[neighbor_index]
                neighbor_weights = self.sample_weights[neighbor_index]

                synthetic_genotype = self._vectorized_sample_hybrid(
                    genotype, neighbor_genotype
                )
                synthetic_coords = (self.y[i] + neighbor_coords) / 2
                synthetic_weights = (self.sample_weights[i] + neighbor_weights) / 2

                over_sampled_X_list.append(synthetic_genotype)
                over_sampled_y_list.append(synthetic_coords)
                over_sampled_S_list.append(synthetic_weights)
                parent_list.append(i)
                mate_list.append(int(neighbor_index))
                sample_origin_list.append("synthetic")  # Mark as synthetic

        # Convert lists to numpy arrays
        over_sampled_X = np.array(over_sampled_X_list)
        over_sampled_y = np.array(over_sampled_y_list)
        over_sampled_S = np.array(over_sampled_S_list)
        self.sample_origin_list = np.array(sample_origin_list)
        self.parent_indices = np.array(parent_list)
        self.mate_indices = np.array(mate_list)

        self.cluster_labels = self._assign_labels_to_synthetic_samples(
            original_labels, over_sampled_y, self.y
        )

        over_sampled_X, over_sampled_y, over_sampled_S = self._shuffle_over_sampled(
            over_sampled_X, over_sampled_y, over_sampled_S
        )

        if self.verbose >= 1:
            n_syn = int(np.sum(self.sample_origin_list == "synthetic"))
            self.logger.info(
                f"Oversampling factors per cluster: {oversample_factors}; added {n_syn} synthetic samples to {len(self.X)} original samples."
            )

        return over_sampled_X, over_sampled_y, over_sampled_S

    def _shuffle_over_sampled(self, over_sampled_X, over_sampled_y, over_sampled_S):
        """Shuffle the over-sampled data.

        The same permutation is applied to ``cluster_labels``, ``sample_origin_list``, ``parent_indices``, and ``mate_indices`` so they stay aligned with the returned rows.

        Args:
            over_sampled_X (np.ndarray): Over-sampled genotype array.
            over_sampled_y (np.ndarray): Over-sampled geographic coordinates.
            over_sampled_S (np.ndarray): Over-sampled sample weights.

        Returns:
            tuple: A tuple containing the shuffled over-sampled genotype array, geographic coordinates, and sample weights.
        """
        shuffled = self.rng.permutation(over_sampled_X.shape[0])
        over_sampled_X = over_sampled_X[shuffled]
        over_sampled_y = over_sampled_y[shuffled]
        over_sampled_S = over_sampled_S[shuffled]

        for attr in (
            "cluster_labels",
            "sample_origin_list",
            "parent_indices",
            "mate_indices",
        ):
            values = getattr(self, attr, None)
            if values is not None and len(values) == len(shuffled):
                setattr(self, attr, np.asarray(values)[shuffled])

        if len(over_sampled_X.shape) == 3:
            over_sampled_X = np.squeeze(over_sampled_X)
        if len(over_sampled_y.shape) > 2:
            over_sampled_y = np.squeeze(over_sampled_y)
        if len(over_sampled_S.shape) > 1:
            over_sampled_S = np.squeeze(over_sampled_S)

        return (over_sampled_X, over_sampled_y, over_sampled_S)

    def _estimate_allele_frequencies(self, genotypes):
        """Estimate allele frequencies from a genotype array.

        This method estimates allele frequencies from a genotype array.

        Args:
            genotypes (np.ndarray): A numpy array of genotypes (0, 1, 2).

        Returns:
            tuple: A tuple containing the frequency of the reference allele and the frequency of the alternate allele.
        """
        count_0 = np.sum(genotypes == 0)  # Homozygous reference
        count_1 = np.sum(genotypes == 1)  # Heterozygous
        count_2 = np.sum(genotypes == 2)  # Homozygous alternate

        total_alleles = 2 * len(genotypes)
        freq_ref = (2 * count_0 + count_1) / total_alleles
        freq_alt = (2 * count_2 + count_1) / total_alleles

        return freq_ref, freq_alt

    def _vectorized_sample_hybrid(
        self,
        genotype1,
        genotype2,
        sampling_strategy="mendelian",
        prob_distributions=None,
    ):
        """Vectorized method to sample hybrid genotype based on two parent genotypes with configurable probabilities.

        Args:
            genotype1 (np.ndarray): The first genotype array.
            genotype2 (np.ndarray): The second genotype array.
            sampling_strategy (str): The strategy for sampling at loci where the 0/1/2 parent genotypes differ ('mendelian', 'hets_only', 'no_hets'). Positions with other (continuous) values get the parents' midpoint.
            prob_distributions (dict, optional): Dictionary containing probability distributions for different parent genotypes.

        Returns:
            np.ndarray: Hybrid genotype array.

        Raises:
            AssertionError: If the genotype arrays are not of the same length.
            ValueError: If an invalid sampling strategy is provided.
        """
        # Ensure that the genotype arrays are of the same length
        if len(genotype1) != len(genotype2):
            msg = "Genotype arrays must be of the same length"
            self.logger.error(msg)
            raise AssertionError(msg)

        if sampling_strategy not in {"mendelian", "no_hets", "hets_only"}:
            msg = f"Invalid 'sampling_strategy' argument provided. Supported options include 'mendelian', 'no_hets', or 'hets_only', but got: {sampling_strategy}."
            self.logger.error(msg)
            raise ValueError(msg)

        genotype1 = np.asarray(genotype1)
        genotype2 = np.asarray(genotype2)

        # Start from an all-NaN float array and track which positions are set.
        interp_genotype = np.full(
            genotype1.shape,
            np.nan,
            dtype=np.result_type(genotype1.dtype, genotype2.dtype, np.float32),
        )
        assigned = np.zeros(genotype1.shape, dtype=bool)

        # Positions where both parents have a 0/1/2 genotype.
        is_012 = np.isin(genotype1, (0, 1, 2)) & np.isin(genotype2, (0, 1, 2))

        # Case when both parents have the same genotype
        same_genotype = genotype1 == genotype2
        interp_genotype[same_genotype] = genotype1[same_genotype]
        assigned |= same_genotype

        # Continuous (e.g., embedded) or other non-0/1/2 values: use the
        # parents' midpoint, matching the synthetic coordinates.
        continuous = ~is_012 & ~same_genotype
        if continuous.any():
            if not self._warned_continuous:
                self.logger.warning(
                    "Features contain values other than 0/1/2 (e.g., embedded features); synthetic values at those positions are the midpoint of the two parents."
                )
                self._warned_continuous = True
            interp_genotype[continuous] = (
                genotype1[continuous] + genotype2[continuous]
            ) / 2
            assigned |= continuous

        # Mendelian probabilities if none are provided
        if prob_distributions is None:
            prob_distributions = {
                "AA_AB": [0.5, 0.5, 0.0],  # Probabilities for AA x AB
                "AB_BB": [0.0, 0.5, 0.5],  # Probabilities for AB x BB
                "AA_BB": [0.0, 1.0, 0.0],  # Probabilities for AA x BB
                "AB_AB": [0.25, 0.5, 0.25],  # Probabilities for AB x AB
            }

        differ = is_012 & ~same_genotype
        if sampling_strategy == "no_hets":
            # Estimate allele frequencies from the parent genotypes
            freq_ref, freq_alt = self._estimate_allele_frequencies(
                np.concatenate([genotype1[is_012], genotype2[is_012]])
            )
            interp_genotype[differ] = self.rng.choice(
                [0, 2], size=np.sum(differ), p=[freq_ref, freq_alt]
            )
            assigned |= differ
        elif sampling_strategy == "hets_only":
            interp_genotype[differ] = 1
            assigned |= differ
        else:
            # Default behavior
            # Case for AA (0) x AB (1) and AB (1) x AA (0)
            aa_ab = ((genotype1 == 0) & (genotype2 == 1)) | (
                (genotype1 == 1) & (genotype2 == 0)
            )
            interp_genotype[aa_ab] = self.rng.choice(
                [0, 1, 2], size=np.sum(aa_ab), p=prob_distributions["AA_AB"]
            )

            # Case for AB (1) x BB (2) and BB (2) x AB (1)
            ab_bb = ((genotype1 == 1) & (genotype2 == 2)) | (
                (genotype1 == 2) & (genotype2 == 1)
            )
            interp_genotype[ab_bb] = self.rng.choice(
                [0, 1, 2], size=np.sum(ab_bb), p=prob_distributions["AB_BB"]
            )

            # Case for AA (0) x BB (2) and BB (2) x AA (0)
            aa_bb = ((genotype1 == 0) & (genotype2 == 2)) | (
                (genotype1 == 2) & (genotype2 == 0)
            )
            interp_genotype[aa_bb] = self.rng.choice(
                [0, 1, 2], size=np.sum(aa_bb), p=prob_distributions["AA_BB"]
            )

            # Case for AB (1) x AB (1)
            ab_ab = (genotype1 == 1) & (genotype2 == 1)
            interp_genotype[ab_ab] = self.rng.choice(
                [0, 1, 2], size=np.sum(ab_ab), p=prob_distributions["AB_AB"]
            )
            assigned |= aa_ab | ab_bb | aa_bb | ab_ab

        if not assigned.all():
            msg = "Some hybrid genotype positions were not assigned."
            self.logger.error(msg)
            raise AssertionError(msg)

        return interp_genotype

    def _assign_labels_to_synthetic_samples(
        self, original_labels, over_sampled_y, original_y
    ):
        """Assign cluster labels to synthetic samples based on the nearest original samples.

        This method assigns cluster labels to synthetic samples based on the nearest original samples.

        Args:
            original_labels (np.ndarray): Cluster labels for the original samples.
            over_sampled_y (np.ndarray): Over-sampled geographic coordinates.
            original_y (np.ndarray): Original geographic coordinates.

        Returns:
            np.ndarray: Cluster labels for the synthetic samples.
        """
        # Fit NearestNeighbors on the original data points
        nbrs = NearestNeighbors(n_neighbors=1).fit(original_y)

        # Find the nearest original point for each synthetic point
        _, indices = nbrs.kneighbors(over_sampled_y)

        # Propagate the labels from the nearest original points to the synthetic points
        synthetic_labels = original_labels[indices.flatten()]

        return synthetic_labels

    def _perform_kmeans_clustering(self):
        """Perform KMeans clustering on the embedded genotypes.

        This method performs KMeans clustering on the embedded genotypes.

        Returns:
            np.ndarray: Cluster labels for each sample.
        """
        sils = []
        cluster_range = range(3, 11)

        best_k = self.n_bins
        if self.n_bins is None:
            for k in cluster_range:
                km = KMeans(
                    n_clusters=k,
                    n_init=25,
                    max_iter=1000,
                    random_state=sklearn_seed(self.rng),
                )
                km.fit(self.y)
                sil = silhouette_score(
                    self._latlon_radians(), km.labels_, metric="haversine"
                )
                sils.append(sil)

            best_k = list(cluster_range)[np.argmax(sils)]
            self.logger.info(f"Best KMeans Silhouette score: {np.max(sils)}")
            self.logger.info(
                f"Optimal clusters for KMeans density estimation: {best_k}"
            )
        km = KMeans(
            n_clusters=best_k,
            n_init=25,
            max_iter=1000,
            random_state=sklearn_seed(self.rng),
        )
        km.fit(self.y)

        return km.labels_, km.cluster_centers_

    def _latlon_radians(self):
        """Return coordinates as (latitude, longitude) in radians, as sklearn's haversine metric expects."""
        return np.radians(np.asarray(self.y, dtype=float)[:, ::-1])

    def _calculate_centroids(self):
        """Calculate centroids of the embedded genotypes using the median of the clusters.

        Returns:
            np.ndarray: Centroids of the genotypes.
        """
        unique_clusters = np.unique(self.cluster_labels)
        centroids = np.array(
            [
                np.median(self.X[self.cluster_labels == cluster], axis=0)
                for cluster in unique_clusters
            ]
        )

        return centroids

    def _calculate_centroid_genotype(self, centroid_index):
        """Calculate the genotype of a given centroid index.

        This method calculates the genotype of a given centroid index.

        Args:
            centroid_index (int): Index of the centroid.

        Returns:
            np.ndarray: Genotype of the centroid.
        """
        # Get the genotype indices that belong to the given cluster
        genotype_indices = np.where(self.cluster_labels == centroid_index)[0]

        # Calculate the median genotype for this centroid
        centroid_genotype = np.median(self.X[genotype_indices], axis=0)

        return centroid_genotype

    def _calculate_optimal_bandwidth(self):
        """Calculate the optimal bandwidth for KernelDensity using grid search.

        The KDE uses the haversine metric on (latitude, longitude) in radians; the bandwidth grid spans 0.01 to 10 degrees of great-circle arc (about 1.1 to 1100 km).

        Returns:
            float: The optimal bandwidth, in radians.
        """
        grid = GridSearchCV(
            KernelDensity(metric="haversine", kernel="gaussian"),
            param_grid={
                "bandwidth": np.radians(np.logspace(-2, 1, num=50, endpoint=True)),
            },
            n_jobs=-1,
            cv=4,
            verbose=False,
        )
        grid.fit(self._latlon_radians())

        self.kde = grid.best_estimator_
        return grid.best_params_["bandwidth"]

    def _perform_density_estimation(self):
        """Perform density estimation using the pre-fitted KernelDensity estimator.

        Returns:
            np.ndarray: Cluster labels for each sample based on KernelDensity estimation.
        """
        # Check if KernelDensity estimator is already fitted
        if self.kde is None:
            msg = "KernelDensity estimator did not get fitted."
            self.logger.error(msg)
            raise AttributeError(msg)

        # Compute log density scores for each sample using the pre-fitted
        # estimator
        log_densities = self.kde.score_samples(self._latlon_radians())
        densities = np.exp(log_densities)  # Convert to actual densities

        # Assign cluster labels based on the threshold
        cluster_labels = np.where(densities >= self.threshold, 1, 0)

        return cluster_labels

    def _automated_parameter_tuning(self, metric=silhouette_score):
        """Automatically tune parameters for the Kerneldensity bandwidth and density threshold.

        Thresholds are searched between the 5th and 45th density percentiles, so the low-density group (label 0) is always the smaller group and is the one that gets oversampled.

        Args:
            metric (callable): Clustering evaluation metric function. Defaults to ``silhouette_score``.

        Returns:
            tuple: The best bandwidth and threshold value combination.
        """
        best_score = -1
        best_params = (None, None)

        bandwidth = self._calculate_optimal_bandwidth()

        if self.kde is None:
            msg = "KernelDensity object is not fitted."
            self.logger.error(msg)
            raise AttributeError(msg)

        # Compute log density scores for each sample
        coords = self._latlon_radians()
        log_densities = self.kde.score_samples(coords)
        densities = np.exp(log_densities)  # Convert to actual densities

        # Experiment with different ways to set the threshold. Only
        # percentiles below the median, so the sparse group is the minority.
        for percentile in range(5, 50, 5):
            threshold = np.percentile(densities, percentile)
            self.threshold = threshold
            cluster_labels = self._perform_density_estimation()

            if (cluster_labels == cluster_labels[0]).all():
                continue

            score = metric(coords, cluster_labels, metric="haversine")
            if score > best_score:
                best_score = score
                best_params = (bandwidth, threshold)

        return best_params


logger = logging.getLogger(__name__)


def run_genotype_interpolator(train_loader, args, ds, dtype, plotting, seed=None):
    """Interpolate genotypes and geographic coordinates among nearest neighbors.

    This method interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        args (argparse.Namespace): User-supplied (or default) arguments.
        ds (geogenie.data.dataset.Dataset): The dataset object.
        dtype (torch.dtype): The data type of the model.
        plotting (geogenie.plotting.plotting.PlotGenIE): PlotGenIE object for visualization.
        seed (int, np.random.Generator, or None, optional): Seed for the synthetic samples. If None, uses ``args.seed``. Defaults to None.

    Returns:
        tuple: A tuple containing the training data loader, centroids, features, labels, and sample weights.
    """
    (centroids, gi, features, labels, sample_weights, indices) = resample_interp(
        train_loader, args, ds, seed=seed
    )

    if features is None or labels is None:
        msg = "Synthetic data augmentation failed. Try adjusting the parameters supplied to GenotypeInterpolator."
        logger.error(msg)
        raise ValueError(msg)

    plotting.visualize_oversample_clusters(
        ds.norm.inverse_transform(labels), gi.cluster_labels, gi.sample_origin_list
    )

    train_loader, features, labels, sample_weights = process_interp(
        train_loader, features, labels, sample_weights, indices, args, ds, dtype
    )

    return train_loader, centroids, features, labels, sample_weights


def process_interp(
    train_loader, features, labels, sample_weights, indices, args, ds, dtype
):
    """Process the interpolated data for training.

    This method processes the interpolated data for training.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        features (np.ndarray): The interpolated features.
        labels (np.ndarray): The interpolated labels.
        sample_weights (np.ndarray): The sample weights.
        indices (np.ndarray): The indices of the samples.
        args (argparse.Namespace): User-supplied (or default) arguments.
        ds (geogenie.data.dataset.Dataset): The dataset object.
        dtype (torch.dtype): The data type of the model.

    Returns:
        tuple: A tuple containing the training data loader, features, labels, and sample weights.
    """
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=dtype)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=dtype)
    if not isinstance(sample_weights, torch.Tensor):
        sample_weights = torch.tensor(sample_weights, dtype=dtype)

    train_dataset = CustomDataset(
        features, labels, sample_weights=sample_weights, dtype=dtype
    )

    kwargs = {"batch_size": train_loader.batch_size}

    if args.use_weighted == "none":
        sample_weights = torch.ones_like(sample_weights, dtype=dtype)
        kwargs["shuffle"] = True
    else:
        if args.use_weighted in ["sampler", "both"]:
            kwargs = reset_weighted_sampler(sample_weights, kwargs, indices, ds)

        if args.use_weighted in ["loss", "both"]:
            if not isinstance(sample_weights, torch.Tensor):
                sample_weights = torch.tensor(sample_weights, dtype=dtype)

            if args.use_weighted != "both":
                kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **kwargs)
    train_loader.dataset.sample_weights = sample_weights
    return train_loader, features, labels, sample_weights


def resample_interp(train_loader, args, ds, seed=None):
    """Resample and interpolate genotypes and geographic coordinates among nearest neighbors.

    This method resamples and interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

    Settings read from ``args``: ``oversample_method``, ``n_bins``, ``oversample_neighbors`` (number of neighbors; None uses the square root of the number of samples), ``max_oversample_factor`` (default 5 if absent), ``seed``, and, for the weights, ``use_weighted`` and the GeographicDensitySampler options.

    Sample weights: when ``args.use_weighted != "none"``, the density weights are recomputed on the oversampled training set (original plus synthetic rows) with the same GeographicDensitySampler settings, so oversampling and inverse-density weighting are not applied twice to the same imbalance.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        args (argparse.Namespace): User-supplied (or default) arguments.
        ds (geogenie.data.dataset.Dataset): The dataset object.
        seed (int, np.random.Generator, or None, optional): Seed for the synthetic samples. If None, uses ``args.seed``. Defaults to None.

    Returns:
        tuple: A tuple containing the centroids, GenotypeInterpolator object, features, labels, sample weights, and indices.
    """
    kdtree = False if args.oversample_method == "kmeans" else True
    if seed is None:
        seed = getattr(args, "seed", None)

    gi = GenotypeInterpolator(
        train_loader.dataset.features.numpy(),
        ds.norm.inverse_transform(train_loader.dataset.labels.numpy()),
        train_loader.dataset.sample_weights.numpy(),
        args,
        n_neighbors=getattr(args, "oversample_neighbors", None),
        use_kde_densities=kdtree,
        n_bins=args.n_bins,
        verbose=args.verbose,
        fontsize=args.fontsize,
        dpi=args.plot_dpi,
        plot_type=args.filetype,
        max_oversample_factor=getattr(args, "max_oversample_factor", 5),
        seed=seed,
    )

    features, labels, sample_weights = gi.interpolate_genotypes()

    if getattr(args, "use_weighted", "none") != "none":
        sampler = GeographicDensitySampler(
            pd.DataFrame(np.asarray(labels, dtype=float), columns=["x", "y"]),
            focus_regions=getattr(args, "focus_regions", None),
            use_kmeans=getattr(args, "use_kmeans", False),
            use_kde=getattr(args, "use_kde", False),
            use_dbscan=getattr(args, "use_dbscan", False),
            w_power=getattr(args, "w_power", 1.0),
            max_clusters=getattr(args, "max_clusters", 10),
            max_neighbors=getattr(args, "max_neighbors", 50),
            normalize=getattr(args, "normalize_sample_weights", False),
            verbose=args.verbose,
            logger=logger,
            kde_bandwidth=getattr(args, "kde_bandwidth", None),
            seed=gi.rng,
        )
        sample_weights = sampler.weights
        if args.verbose >= 1:
            logger.info(
                "Recomputed sample weights on the oversampled training set."
            )

    indices = np.arange(features.shape[0])
    centroids = gi.centroids
    labels = ds.norm.transform(labels)

    if centroids is not None:
        centroids = ds.norm.transform(centroids)
    return centroids, gi, features, labels, sample_weights, indices


def reset_weighted_sampler(sample_weights, kwargs, indices, ds):
    """Reset the weighted sampler for the training data loader.

    This method resets the weighted sampler for the training data loader.

    Args:
        sample_weights (np.ndarray): The sample weights.
        kwargs (dict): The keyword arguments for the training data loader.
        indices (np.ndarray): The indices of the samples.
        ds (geogenie.data.dataset.Dataset): The dataset object.

    Returns:
        dict: The updated keyword arguments for the training data loader.
    """

    weighted_sampler = deepcopy(ds.weighted_sampler)
    weighted_sampler.indices = indices

    if isinstance(sample_weights, torch.Tensor):
        sample_weights = sample_weights.numpy()
    weighted_sampler.weights = sample_weights
    kwargs["sampler"] = weighted_sampler
    return kwargs
