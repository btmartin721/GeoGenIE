# GeoGenIE User Manual

![GeoGenIE Logo](img/geogenie_logo.png){width=40%}

## Introduction

GeoGenIE (Geographic-Genetic Inference Engine) is a comprehensive software tool designed to predict geographic coordinates (longitude and latitude) from genetic SNP data. GeoGenIE utilizes deep learning models and offers several advanced features such as automated parameter tuning with Optuna [1], outlier detection to remove individuals that do not conform to expected geographic and genetic patterns (e.g., isolation by distance) [3], handling of imbalanced sampling using a custom oversampling algorithm adapted from SMOTE [2], and weighting the loss function by inverse sampling densities. The software is user-friendly and provides extensive visualizations and metrics for evaluating predictions, making it a robust and accurate solution for geographic-genetic inference.

## Installation

To install GeoGenIE, it is recommended to use a virtual environment or conda environment. From the root project directory, enter the following command while in the root project directory.

```bash
pip install GeoGenIE
```

### Dependencies

The following packages will be installed when running `pip install GeoGenIE`:

- geopandas
- geopy
- imblearn
- jenkspy
- kaleido
- kneed
- matplotlib
- numba
- numpy
- optuna
- pandas
- plotly
- pykrige
- pynndescent
- pysam
- pyyaml
- requests
- scikit-learn
- scipy
- seaborn
- statsmodels
- torch
- xgboost

## Usage

### Running GeoGenIE

GeoGenIE can be run with individual command-line arguments, but using a YAML config file is recommended. See `config_files/config.yaml` for an example YAML file. Assuming GeoGenIE is installed in your environment, you can run it like this:

```bash
geogenie --config config_files/config.yaml
```

### Command-line Options

You can see all the command-line options by running the help flag:

```bash
geogenie -h
```

**Note:** If you don't want to use the configuration file, you can specify each argument individually on the command line. For example:

```bash
geogenie --output_dir "my_analysis" --prefix "my_prefix" <other arguments>
```

### Configuration File

You can set all the options for input files, model parameters, etc., in the `config_files/config.yaml` file. Using a configuration file allows tracking of parameters across multiple runs and ensures better reproducibility.

- Python `None` values are represented by `null` (without quotes).
- Python `True` values are represented by `true` (all lowercase, no quotes).
- Python `False` values are represented by `false` (all lowercase, no quotes).

You can also leave comments with `# my_comment`. The arguments can be in any order in the `config_files/config.yaml` file.

### Running the Software

```bash
geogenie --config config_files/config.yaml
```

| **Option**                | **Description**                                                                                           | **Default**         | **Notes**                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------------------|---------------------|------------------------------------------------------------------------------------------------|
| `--vcf`                   | Path to the VCF file with SNPs.                                                                          | None                | Required input file.                                                                            |
| `--sample_data`           | Path to the coordinates file.                                                                            | None                | Required input file.                                                                            |
| `--known_sample_data`     | Path to the known coordinates file.                                                                      | None                | Used for per-sample bootstrapped output plots.                                                 |
| `--prop_unknowns`         | Proportion of samples set to "unknown."                                                                  | 0.1                 | Useful for testing model prediction.                                                           |
| `--min_mac`               | Minimum minor allele count to retain SNPs.                                                               | 2                   | Filters rare variants.                                                                          |
| `--max_SNPs`              | Maximum number of SNPs to randomly subset.                                                               | None                | Reduces computational load.                                                                    |
| `--embedding_type`        | Embedding type for input SNP dataset.                                                                    | "none"              | Options: 'pca', 'kernelpca', 'nmf', 'lle', 'mca', 'mds', 'polynomial', 'tsne', and 'none'.     |
| `--n_components`          | Number of components for PCA/tSNE embeddings.                                                           | None                | Auto-detected for optimal performance.                                                         |
| `--embedding_sensitivity` | Sensitivity for optimal component selection.                                                             | 1.0                 | Adjust for balance between overfitting and underfitting.                                       |
| `--tsne_perplexity`       | Perplexity setting for T-SNE embedding.                                                                  | 30                  | Lower values for finer details.                                                                |
| `--polynomial_degree`     | Polynomial degree for 'polynomial' embedding.                                                            | 2                   | Higher degrees increase complexity and computational overhead.                                  |
| `--n_init`                | Initialization runs for Multi-Dimensional Scaling embedding.                                             | 4                   | More initialization runs provide stable results but increase computation time.                 |
| `--nlayers`               | Number of hidden layers in the neural network.                                                           | 10                  | Adjust for model complexity.                                                                   |
| `--width`                 | Number of neurons per layer.                                                                             | 256                 | Controls model capacity.                                                                       |
| `--dropout_prop`          | Dropout rate to prevent overfitting.                                                                     | 0.25                | Regularization technique.                                                                      |
| `--criterion`             | Model loss criterion.                                                                                    | "rmse"              | Options: 'rmse', 'huber', 'drms'.                                                              |
| `--load_best_params`      | Load best parameters from previous Optuna search.                                                        | None                | Save time by reusing optimized parameters.                                                     |
| `--use_gradient_boosting` | Use Gradient Boosting model instead of deep learning model.                                              | False               | Deprecated; may be removed in future versions.                                                 |
| `--dtype`                 | PyTorch data type.                                                                                       | "float32"           | Options: 'float32', 'float64'.                                                                 |
| `--batch_size`            | Training batch size.                                                                                     | 32                  | Affects training stability and memory usage.                                                   |
| `--max_epochs`            | Maximum number of training epochs.                                                                       | 5000                | Early stopping will terminate training earlier if conditions are met.                          |
| `--learning_rate`         | Learning rate for the optimizer.                                                                         | 0.001               | Adjust to control training speed.                                                              |
| `--l2_reg`                | L2 regularization weight.                                                                                | 0.0                 | Reduces overfitting.                                                                           |
| `--early_stop_patience`   | Epochs to wait after no improvement before stopping.                                                     | 48                  | Prevents excessive training time when no further improvements are observed.                    |
| `--train_split`           | Proportion of data used for training.                                                                    | 0.8                 | Validation and test splits must sum to 1.0.                                                    |
| `--val_split`             | Proportion of data used for validation.                                                                  | 0.2                 | Adjust based on the dataset size.                                                              |
| `--do_bootstrap`          | Enable bootstrap replicates.                                                                             | False               | Generates confidence intervals for predictions.                                                |
| `--nboots`                | Number of bootstrap replicates.                                                                          | 100                 | Increase for more accurate confidence intervals.                                               |
| `--do_gridsearch`         | Perform Optuna parameter search.                                                                         | False               | Optimizes model parameters for better performance.                                             |
| `--n_iter`                | Iterations for parameter optimization using Optuna.                                                      | 100                 | Higher values result in more thorough parameter searches.                                      |
| `--lr_scheduler_patience` | Learning rate scheduler patience.                                                                        | 16                  | Reduce learning rate after no improvement over specified epochs.                               |
| `--lr_scheduler_factor`   | Factor to reduce learning rate when scheduler is triggered.                                              | 0.5                 | Lower values make finer adjustments to the learning rate.                                      |
| `--factor`                | Scale factor for neural network widths in successive layers.                                             | 1.0                 | Controls width reduction of hidden layers.                                                     |
| `--grad_clip`             | Enable gradient clipping.                                                                                | False               | Prevents gradient explosion in deep networks.                                                  |
| `--use_weighted`          | Use inverse-weighted probability sampling.                                                               | "none"              | Options: 'loss', 'sampler', 'both', or 'none'.                                                 |
| `--oversample_method`     | Synthetic oversampling method.                                                                           | "none"              | Options: 'kmeans', 'none'.                                                                     |
| `--oversample_neighbors`  | Number of nearest neighbors for oversampling.                                                            | 5                   | Controls synthetic sample generation.                                                          |
| `--n_bins`                | Number of bins for synthetic resampling.                                                                 | 8                   | Higher values enable finer-grained sampling density adjustments.                                |
| `--use_kmeans`            | Use KMeans clustering for weighted loss.                                                                 | False               | Weighted loss focuses on undersampled geographic regions.                                      |
| `--use_kde`               | Use Kernel Density Estimation to obtain sample weights.                                                  | False               | Experimental feature.                                                                          |
| `--use_dbscan`            | Use DBSCAN clustering to obtain sample weights.                                                          | False               | Experimental feature.                                                                          |
| `--w_power`               | Exponential power for inverse density weighting.                                                         | 1.0                 | Controls the emphasis on undersampled regions.                                                 |
| `--max_clusters`          | Maximum number of clusters for KMeans.                                                                   | 10                  | Higher values allow more detailed clustering.                                                  |
| `--focus_regions`         | Geographic regions of interest for sampling density weights.                                             | None                | Format: [(lon_min, lon_max, lat_min, lat_max), ...].                                           |
| `--normalize_sample_weights` | Normalize density-based sample weights from 0 to 1.                                                  | False               | Ensures all sample weights are on a comparable scale.                                          |
| `--detect_outliers`       | Enable outlier detection to remove translocated individuals.                                             | False               | Helps to clean the dataset.                                                                    |
| `--min_nn_dist`           | Minimum distance between nearest neighbors for outlier detection (meters).                               | 1000                | Adjust to control outlier detection stringency.                                                |
| `--scale_factor`          | Geographic distance scaling factor for outlier detection.                                                | 100                 | Controls how geographic distances are evaluated.                                               |
| `--significance_level`    | Significance level (alpha) for outlier detection.                                                        | 0.05                | P-values ≤ alpha are removed as outliers.                                                     |
| `--maxk`                  | Maximum number of nearest neighbors for outlier detection.                                               | 50                  | Controls the scope of neighbor comparisons.                                                    |
| `--show_plots`            | Display plots inline (e.g., in Jupyter Notebooks).                                                       | False               | Visualization aid.                                                                             |
| `--fontsize`              | Font size for plot labels, ticks, and titles.                                                            | 24                  | Adjust to improve readability of plots.                                                       |
| `--filetype`              | File type for saving plots.                                                                              | "png"               | Options: 'png', 'pdf', 'jpg'.                                                                  |
| `--plot_dpi`              | DPI for image format plots.                                                                              | 300                 | Higher DPI produces clearer images but increases file size.                                    |
| `--remove_splines`        | Remove bottom and left axis splines from map plots.                                                      | False               | Use for cleaner map visualizations.                                                           |
| `--shapefile`             | URL or file path for shapefile used in plotting.                                                         | Default USA map     | Specify a custom shapefile for regions outside the USA.                                        |
| `--basemap_fips`          | FIPS code for basemap.                                                                                   | None                | Specify to focus on specific states or regions (e.g., "05" for Arkansas).                      |
| `--highlight_basemap_counties` | Highlight specified counties on the base map in gray.                                              | None                | Useful for emphasizing areas of interest.                                                     |
| `--samples_to_plot`       | Comma-separated sample IDs for per-sample bootstrap plots.                                               | None                | Defaults to plotting all samples.                                                             |
| `--n_contour_levels`      | Number of contour levels for the Kriging plot.                                                           | 20                  | Adjust for more detailed or simplified contour plots.                                          |
| `--min_colorscale`        | Minimum colorbar value for the Kriging plot.                                                             | 0                   | Set to match the range of your data.                                                          |
| `--max_colorscale`        | Maximum colorbar value for the Kriging plot.                                                             | 300                 | Increase if your data range exceeds the default.                                              |
| `--sample_point_scale`    | Scale factor for sample point size on Kriging plot.                                                      | 2                   | Adjust to ensure points are visible but not overwhelming.                                      |
| `--bbox_buffer`           | Buffer for the sampling bounding box on map visualizations.                                              | 0.1                 | Increase to add more context around the sampling area.                                        |
| `--prefix`                | Output file prefix.                                                                                      | "output"            | Helps organize output files for multiple runs.                                                |
| `--sqldb`                 | SQLite3 database directory for Optuna optimization.                                                     | None                | Saves optimization results for future use.                                                    |
| `--output_dir`            | Directory to store output files.                                                                         | "./output"          | Organize files by specifying unique output directories.                                        |
| `--seed`                  | Random seed for reproducibility.                                                                         | None                | Ensures consistent results across runs.                                                       |
| `--gpu_number`            | GPU number for computation.                                                                              | None (CPU)          | Specify to enable GPU acceleration. Requires CUDA-compatible hardware.                         |
| `--n_jobs`                | Number of CPU jobs to use.                                                                               | -1                  | Use all CPUs by default.                                                                      |
| `--verbose`               | Verbosity level for logging.                                                                             | 1                   | Options: 0 (silent) to 3 (most verbose).                                                      |
| `--debug`                 | Enable debug mode for verbose logging.                                                                   | False               | Useful for troubleshooting.                                                                    |

## Output Files and File Structure

Outputs are saved to the directory specified by `--output_dir <my_output_dir>/<prefix_>_*`. The prefix is specified with `--prefix <prefix>`. The directory structure of `<output_dir>` includes:

| **Directory**              | **Description**                                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `benchmarking`             | Execution times for model training and prediction, with one line per bootstrap replicate if using bootstrapping.          |
| `bootstrapped_sample_ci`   | One plot per sample showing confidence intervals on a map.                                                                |
| `bootstrap_metrics`        | JSON files with statistical metrics for each bootstrap replicate, in `test` and `val` subdirectories.                     |
| `bootstrap_predictions`    | CSV files containing predictions for each bootstrap replicate, in `test`, `val`, and `unknown` subdirectories.            |
| `bootstrap_summaries`      | Mean, median, and standard deviation representations of bootstrap replicates for the test, val, and unknown ("pred") datasets. |
| `data`                     | Text files with sample IDs detected as outliers if `--detect_outliers` is enabled.                                        |
| `logfiles`                 | Logs with INFO, WARNING, and ERROR messages, including timestamps and GeoGenIE modules.                                   |
| `models`                   | Trained PyTorch models saved as ".pt" files, one per bootstrap if `--do_bootstrap` is enabled.                            |
| `optimize`                 | Optuna results, including the best-found parameters as a JSON file.                                                       |
| `plots`                    | All plots and visualizations, including model prediction error visualizations and the basemap shapefile specified with `--shapefile <url>`. Per-sample plots visualizing bootstrapped prediction error are saved in the `plots/bootstrapped_sample_ci` subdirectory. |

**Warning**: Re-running GeoGenIE with the same `output_dir` and `prefix` will overwrite all outputs except the Optuna SQL database.

## References

[1] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th International Conference on Knowledge Discovery and Data Mining. <https://doi.org/10.1145/3292500.3330701>

[2] Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. Journal of Machine Learning Research, 18(17), 1-5. <http://jmlr.org/papers/v18/16-365>  

[3] Chang et al., (2023). GGoutlieR: an R package to identify and visualize unusual geo-genetic patterns of biological samples. Journal of Open Source Software, 8(91), 5687. <https://doi.org/10.21105/joss.05687>
