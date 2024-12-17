# Module: samplers

## Imports

- logging
- os
- jenkspy
- matplotlib.pyplot
- numpy
- pandas
- seaborn
- torch
- geopy.distance
- imblearn.combine
- imblearn.over_sampling
- imblearn.under_sampling
- scipy
- scipy.spatial.distance
- scipy.stats
- sklearn.cluster
- sklearn.metrics
- sklearn.neighbors
- sklearn.preprocessing
- geogenie.utils.spatial_data_processors
- geogenie.utils.utils
- numpy
- sklearn.cluster
- sklearn.metrics
- sklearn.neighbors
- sklearn.preprocessing

## Functions

- synthetic_resampling
- do_kde_binning
- merge_single_sample_bins
- identify_small_bins
- merge_small_bins
- calculate_centroid_distances
- define_jenks_thresholds
- calculate_bin_centers
- assign_samples_to_bins
- spatial_kde
- define_density_thresholds
- get_kde_bins
- get_centroids
- run_binned_smote
- setup_synth_resampling
- process_bins
- custom_gpr_optimizer
- cluster_minority_samples
- __init__
- _plot_cluster_weights
- calculate_weights
- _calculate_kmeans_weights
- _calculate_kde_weights
- _determine_eps
- _calculate_dbscan_weights
- _adjust_for_focus_regions
- calculate_adaptive_bandwidth
- find_optimal_clusters
- __iter__
- __len__

## Classes and Methods

- GeographicDensitySampler
  - __init__
  - _plot_cluster_weights
  - calculate_weights
  - _calculate_kmeans_weights
  - _calculate_kde_weights
  - _determine_eps
  - _calculate_dbscan_weights
  - _adjust_for_focus_regions
  - calculate_adaptive_bandwidth
  - find_optimal_clusters
  - __iter__
  - __len__