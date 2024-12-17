# Module: interpolate

## Imports

- logging
- copy
- pathlib
- matplotlib.pyplot
- numpy
- seaborn
- torch
- scipy.spatial.distance
- sklearn.cluster
- sklearn.metrics
- sklearn.model_selection
- sklearn.neighbors
- torch.utils.data
- geogenie.plotting.plotting
- geogenie.utils.data

## Functions

- run_genotype_interpolator
- process_interp
- resample_interp
- reset_weighted_sampler
- __init__
- _determine_optimal_neighbors
- _find_nearest_neighbors
- interpolate_genotypes
- _shuffle_over_sampled
- _estimate_allele_frequencies
- _vectorized_sample_hybrid
- _assign_labels_to_synthetic_samples
- _perform_kmeans_clustering
- _calculate_centroids
- _calculate_centroid_genotype
- _calculate_optimal_bandwidth
- _perform_density_estimation
- _automated_parameter_tuning

## Classes and Methods

- GenotypeInterpolator
  - __init__
  - _determine_optimal_neighbors
  - _find_nearest_neighbors
  - interpolate_genotypes
  - _shuffle_over_sampled
  - _estimate_allele_frequencies
  - _vectorized_sample_hybrid
  - _assign_labels_to_synthetic_samples
  - _perform_kmeans_clustering
  - _calculate_centroids
  - _calculate_centroid_genotype
  - _calculate_optimal_bandwidth
  - _perform_density_estimation
  - _automated_parameter_tuning