import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, NearestNeighbors
from torch.utils.data import DataLoader

from geogenie.plotting.plotting import PlotGenIE
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
        neighbor_indices (np.ndarray): Indices of the nearest neighbors for each sample.
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
    ):
        """Initialize the GenotypeInterpolator with genotype data and geographic coordinates.

        This class interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

        Args:
            X (np.ndarray): 2D array of genotypes.
            y (np.ndarray): 2D array of normalized geographic coordinates.
            sample_weights (np.ndarray): 1D array of sample weights.
            args (argparse.Namespace): User-supplied (or default) arguments.
            n_neighbors (int, optional): Number of neighbors to consider for each sample. If None, then n_neighbors is estimated from the data. Defaults to None.
            embedded (bool): Indicates if the genotypes are embedded (dimensionality reduced). Defaults to False.
            use_kde_densities (bool): Flag to use KernelDensity for density estimation in clustering. Defaults to False.
            n_bins (int or None): Number of clusters to use with KMeans clustsering. If None, then the number of bins are estimated from the data. Defaults to None.
            verbose (int): Verbosity setting (0-3). Defaults to 0 (silent).
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

        self.logger = logging.getLogger(__name__)

        if n_neighbors is not None:
            if n_neighbors not in ["sqrt", "log2"] and not isinstance(n_neighbors, int):
                msg = f"'n_neighbors' must be either 'sqrt', 'log2', NoneType, or an integer, but got: {n_neighbors}."
                self.logger.error(msg)
                raise ValueError(msg)
            elif isinstance(n_neighbors, str):
                n_neighbors = n_neighbors.lower()
            else:
                if n_neighbors <= 1:
                    msg = f"'n_neighbors must be > 1, but got: {n_neighbors}"
                    self.logger.error(msg)
                    raise ValueError(msg)
                self.n_neighbors = n_neighbors

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
            n_neighbors (int or str): Number of neighbors to consider for each sample.

        Returns:
            int: Estimated optimal number of neighbors.

        Raises:
            TypeError: If an invalid type is provided for n_neighbors.
        """
        if isinstance(n_neighbors, str):
            method = np.sqrt if n_neighbors == "sqrt" else np.log2
        else:
            msg = f"Expected str, but got: {type(n_neighbors)}"
            self.logger.error(msg)
            raise TypeError(msg)

        # Simple heuristic: square root of the number of samples
        optimal_n = int(method(len(self.y)))
        return optimal_n

    def _find_nearest_neighbors(self):
        """Find the nearest neighbors or centroids for each sample based on geographic coordinates.

        Returns:
            np.ndarray: Indices of the nearest neighbors for each sample.
        """
        if self.embedded:
            # Calculate distances to centroids for embedded genotypes
            centroids = self._calculate_centroids()
            distances = cdist(self.y, centroids)
            indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        else:
            # Standard nearest neighbors for non-embedded genotypes
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm="auto")
            nbrs.fit(self.y)
            _, indices = nbrs.kneighbors(self.y)

        return indices

    def interpolate_genotypes(self):
        """Perform synthetic over-sampling of the data by interpolating genotypes and geographic coordinates among nearest neighbors.

        This method performs synthetic over-sampling of the data by interpolating genotypes and geographic coordinates among nearest neighbors.

        Returns:
            np.ndarray: Synthetically over-sampled genotype array and corresponding geographic coordinates.
        """

        # Identify the number of samples in each cluster
        unique_clusters, counts = np.unique(self.cluster_labels, return_counts=True)
        largest_cluster_size = counts.max()

        # Calculate the oversampling factor for each cluster
        oversample_factors = {
            cluster: largest_cluster_size // count
            for cluster, count in zip(unique_clusters, counts)
        }

        # Lists to store over-sampled data and metadata
        over_sampled_X_list = []
        over_sampled_y_list = []
        over_sampled_S_list = []
        neighbor_indices_list = []
        num_synthetic_samples_list = []
        self.sample_origin_list = []  # List to keep track of original and synthetic

        for i, genotype in enumerate(self.X):
            cluster = self.cluster_labels[i]
            num_samples_to_generate = oversample_factors.get(
                cluster, 1
            )  # Get the oversampling factor for the current cluster

            # Choose a neighbor index for synthetic sample generation
            neighbor_index = np.random.choice(self.neighbor_indices[i])

            # Append to lists outside the inner loop
            neighbor_indices_list.append(neighbor_index)
            num_synthetic_samples_list.append(num_samples_to_generate)

            # Add the original sample to the lists
            over_sampled_X_list.append(genotype)
            over_sampled_y_list.append(self.y[i])
            over_sampled_S_list.append(self.sample_weights[i])
            self.sample_origin_list.append("original")  # Mark as original

            # Generate the synthetic samples
            for _ in range(
                num_samples_to_generate - 1
            ):  # Subtract 1 to account for the original sample
                neighbor_genotype = self.X[neighbor_index]
                neighbor_coords = self.y[neighbor_index]
                neighbor_weights = self.sample_weights[neighbor_index]

                if self.embedded:
                    centroid_index = self.neighbor_indices[i][0]  # Closest centroid
                    centroid_genotype = self._calculate_centroid_genotype(
                        centroid_index
                    )
                    centroid_coords = self.X_centroids[centroid_index]
                    centroid_weights = self.sample_weights[centroid_index]

                    synthetic_genotype = self._vectorized_sample_hybrid(
                        genotype, centroid_genotype
                    )
                    synthetic_coords = (self.y[i] + centroid_coords) / 2
                    synthetic_weights = (self.sample_weights[i] + centroid_weights) / 2
                else:
                    synthetic_genotype = self._vectorized_sample_hybrid(
                        genotype, neighbor_genotype
                    )
                    synthetic_coords = (self.y[i] + neighbor_coords) / 2
                    synthetic_weights = (self.sample_weights[i] + neighbor_weights) / 2

                over_sampled_X_list.append(synthetic_genotype)
                over_sampled_y_list.append(synthetic_coords)
                over_sampled_S_list.append(synthetic_weights)
                self.sample_origin_list.append("synthetic")  # Mark as synthetic

        # Convert lists to numpy arrays
        over_sampled_X = np.array(over_sampled_X_list)
        over_sampled_y = np.array(over_sampled_y_list)
        over_sampled_S = np.array(over_sampled_S_list)

        self.cluster_labels = self._assign_labels_to_synthetic_samples(
            self.cluster_labels, over_sampled_y, self.y
        )

        over_sampled_X, over_sampled_y, over_sampled_S = self._shuffle_over_sampled(
            over_sampled_X, over_sampled_y, over_sampled_S
        )

        return over_sampled_X, over_sampled_y, over_sampled_S

    def _shuffle_over_sampled(self, over_sampled_X, over_sampled_y, over_sampled_S):
        """Shuffle the over-sampled data.

        Args:
            over_sampled_X (np.ndarray): Over-sampled genotype array.
            over_sampled_y (np.ndarray): Over-sampled geographic coordinates.
            over_sampled_S (np.ndarray): Over-sampled sample weights.

        Returns:
            tuple: A tuple containing the shuffled over-sampled genotype array, geographic coordinates, and sample weights.
        """
        oversamp_indices = np.arange(over_sampled_X.shape[0])
        shuffled = np.random.shuffle(oversamp_indices)
        over_sampled_X = over_sampled_X[shuffled]
        over_sampled_y = over_sampled_y[shuffled]
        over_sampled_S = over_sampled_S[shuffled]

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
            sampling_strategy (str): The strategy for sampling ('mendelian', 'heterozygous_only', 'no_heterozygous').
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

        # Create an empty array to store the hybrid genotype
        interp_genotype = np.empty_like(genotype1)

        # Case when both parents have the same genotype
        same_genotype = genotype1 == genotype2
        interp_genotype[same_genotype] = genotype1[same_genotype]

        # Mendelian probabilities if none are provided
        if prob_distributions is None:
            prob_distributions = {
                "AA_AB": [0.5, 0.5, 0.0],  # Probabilities for AA x AB
                "AB_BB": [0.0, 0.5, 0.5],  # Probabilities for AB x BB
                "AA_BB": [0.0, 1.0, 0.0],  # Probabilities for AA x BB
                "AB_AB": [0.25, 0.5, 0.25],  # Probabilities for AB x AB
            }

        if sampling_strategy == "no_hets":
            # Estimate allele frequencies from the parent genotypes
            freq_ref, freq_alt = self.estimate_allele_frequencies(
                np.concatenate([genotype1, genotype2])
            )
            interp_genotype[~same_genotype] = np.random.choice(
                [0, 2], size=np.sum(~same_genotype), p=[freq_ref, freq_alt]
            )
        elif sampling_strategy == "hets_only":
            interp_genotype[~same_genotype] = 1
        else:
            # Default behavior
            # Case for AA (0) x AB (1) and AB (1) x AA (0)
            aa_ab = ((genotype1 == 0) & (genotype2 == 1)) | (
                (genotype1 == 1) & (genotype2 == 0)
            )
            interp_genotype[aa_ab] = np.random.choice(
                [0, 1, 2], size=np.sum(aa_ab), p=prob_distributions["AA_AB"]
            )

            # Case for AB (1) x BB (2) and BB (2) x AB (1)
            ab_bb = ((genotype1 == 1) & (genotype2 == 2)) | (
                (genotype1 == 2) & (genotype2 == 1)
            )
            interp_genotype[ab_bb] = np.random.choice(
                [0, 1, 2], size=np.sum(ab_bb), p=prob_distributions["AB_BB"]
            )

            # Case for AA (0) x BB (2) and BB (2) x AA (0)
            aa_bb = ((genotype1 == 0) & (genotype2 == 2)) | (
                (genotype1 == 2) & (genotype2 == 0)
            )
            interp_genotype[aa_bb] = np.random.choice(
                [0, 1, 2], size=np.sum(aa_bb), p=prob_distributions["AA_BB"]
            )

            # Case for AB (1) x AB (1)
            ab_ab = (genotype1 == 1) & (genotype2 == 1)
            interp_genotype[ab_ab] = np.random.choice(
                [0, 1, 2], size=np.sum(ab_ab), p=prob_distributions["AB_AB"]
            )

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
                km = KMeans(n_clusters=k, n_init=25, max_iter=1000)
                km.fit(self.y)
                sil = silhouette_score(self.y, km.labels_, metric="haversine")
                sils.append(sil)

            best_k = list(cluster_range)[np.argmax(sils)]
            self.logger.info(f"Best KMeans Silhouette score: {np.max(sils)}")
            self.logger.info(
                f"Optimal clusters for KMeans density estimation: {best_k}"
            )
        km = KMeans(n_clusters=best_k, n_init=25, max_iter=1000)
        km.fit(self.y)

        return km.labels_, km.cluster_centers_

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

        Returns:
            float: The optimal bandwidth.
        """
        grid = GridSearchCV(
            KernelDensity(metric="haversine", kernel="gaussian"),
            param_grid={
                "bandwidth": np.logspace(-2, 1, num=50, endpoint=True),
            },
            n_jobs=-1,
            cv=4,
            verbose=False,
        )
        grid.fit(self.y)

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
        log_densities = self.kde.score_samples(self.y)
        densities = np.exp(log_densities)  # Convert to actual densities

        # Assign cluster labels based on the threshold
        cluster_labels = np.where(densities >= self.threshold, 1, 0)

        return cluster_labels

    def _automated_parameter_tuning(self, metric=silhouette_score):
        """Automatically tune parameters for the Kerneldensity bandwidth and density threshold.

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
        log_densities = self.kde.score_samples(self.y)
        densities = np.exp(log_densities)  # Convert to actual densities

        # Experiment with different ways to set the threshold
        for percentile in range(5, 90, 5):
            threshold = np.percentile(densities, percentile)
            self.threshold = threshold
            cluster_labels = self._perform_density_estimation()

            if (cluster_labels == cluster_labels[0]).all():
                continue

            score = metric(self.y, cluster_labels)
            if score > best_score:
                best_score = score
                best_params = (bandwidth, threshold)

        return best_params


logger = logging.getLogger(__name__)


def run_genotype_interpolator(train_loader, args, ds, dtype, plotting):
    """Interpolate genotypes and geographic coordinates among nearest neighbors.

    This method interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        args (argparse.Namespace): User-supplied (or default) arguments.
        ds (geogenie.data.dataset.Dataset): The dataset object.
        dtype (torch.dtype): The data type of the model.
        plotting (geogenie.plotting.plotting.PlotGenIE): PlotGenIE object for visualization.

    Returns:
        tuple: A tuple containing the training data loader, centroids, features, labels, and sample weights.
    """
    (centroids, gi, features, labels, sample_weights, indices) = resample_interp(
        train_loader, args, ds
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


def resample_interp(train_loader, args, ds):
    """Resample and interpolate genotypes and geographic coordinates among nearest neighbors.

    This method resamples and interpolates genotypes and geographic coordinates among nearest neighbors to perform synthetic over-sampling.

    Args:
        train_loader (torch.utils.data.DataLoader): The training data loader.
        args (argparse.Namespace): User-supplied (or default) arguments.
        ds (geogenie.data.dataset.Dataset): The dataset object.

    Returns:
        tuple: A tuple containing the centroids, GenotypeInterpolator object, features, labels, sample weights, and indices.
    """
    kdtree = False if args.oversample_method == "kmeans" else True

    gi = GenotypeInterpolator(
        train_loader.dataset.features.numpy(),
        ds.norm.inverse_transform(train_loader.dataset.labels.numpy()),
        train_loader.dataset.sample_weights.numpy(),
        args,
        use_kde_densities=kdtree,
        n_bins=args.n_bins,
        verbose=args.verbose,
        fontsize=args.fontsize,
        dpi=args.plot_dpi,
        plot_type=args.filetype,
    )

    features, labels, sample_weights = gi.interpolate_genotypes()
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
