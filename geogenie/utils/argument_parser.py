import argparse
import ast
import logging
import os
import warnings

import yaml
from torch.cuda import is_available

from geogenie.utils.exceptions import GPUUnavailableError, ResourceAllocationError

logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load the YAML configuration file.

    Args:
        config_path (str): Path to configuration file.

    Returns:
        dict: Configuration arguments.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class EvaluateAction(argparse.Action):
    """Custom action for evaluating complex arguments as Python literal structures."""

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            result = ast.literal_eval(values)
            setattr(namespace, self.dest, result)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Couldn't parse '{values}' as a Python literal."
            )


def validate_positive_int(value):
    """Validate that the provided value is a positive integer."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer.")
    return ivalue


def validate_positive_float(value):
    """Validate that the provided value is a positive float."""
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} must be a positive float.")
    return fvalue


def validate_gpu_number(value):
    """
    Validate the provided GPU number.

    Args:
        value (str): The GPU number provided as a command-line argument.

    Returns:
        int: The validated GPU number.

    Raises:
        argparse.ArgumentTypeError: If the GPU number is invalid.
    """
    # Placeholder validation logic - replace with actual checks as needed
    if value is not None:
        try:
            gpu_number = int(value)
            if gpu_number < 0:
                raise ValueError("'gpu_number' must be >= 0")
        except Exception:
            raise argparse.ArgumentTypeError(f"{value} is not a valid GPU number.")
        if not is_available():
            raise GPUUnavailableError(f"Specified GPU {gpu_number} is not available.")

    else:
        gpu_number = value  # None, if no GPU is used
    return str(gpu_number)


def validate_n_jobs(value):
    """Validate the provided n_jobs parameter.

    Args:
        value (int): the number of jobs to use.

    Returns:
        int: The validated n_jobs parameter.
    """
    try:
        n_jobs = int(value)
        if n_jobs == 0 or n_jobs < -1:
            raise ResourceAllocationError(
                f"'n_jobs' must be > 0 or -1, but got {n_jobs}"
            )
    except ValueError:
        raise ResourceAllocationError(
            f"Invalid 'n_jobs' parameter provided: {n_jobs}; parameter must be > 0 or -1."
        )
    return n_jobs


def validate_split(value):
    try:
        split = float(value)
        if split <= 0.0 or split >= 1.0:
            raise ValueError(
                f"'train_split' and 'val_split' must be > 0.0 and < 1.0: {split}"
            )
    except ValueError:
        raise ValueError(
            f"'train_split' and 'val_split' must be > 0 and < 1.0: {split}"
        )
    return split


def validate_verbosity(value):
    try:
        verb = int(value)
        if verb < 0 or verb > 3:
            raise ValueError(f"'verbose' must >= 0 and <= 3: {verb}")
    except ValueError:
        raise ValueError(f"'verbose' must >= 0 and <= 3: {value}")
    return verb


def validate_seed(value):
    if value is not None:
        try:
            seed = int(value)
            if seed <= 0:
                raise ValueError(f"'seed' must > 0: {seed}")
        except ValueError:
            raise ValueError(f"'seed' must > 0: {seed}")
    else:
        return None
    return seed


def validate_lower_str(value):
    try:
        value = str(value)
    except TypeError:
        raise TypeError(f"Could not convert {value} to a string.")
    return value.lower()


def setup_parser(test_mode=False):
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog="GeoGenIE",
        description="Predict geographic coordinates from genome-wide SNPs using deep learning.",
    )

    # Optional argument for the configuration file
    parser.add_argument(
        "--config", type=str, help="Path to the configuration YAML file."
    )

    # Data Input and Preprocessing Arguments
    aln_group = parser.add_mutually_exclusive_group()
    aln_group.add_argument(
        "--vcf",
        default=None,
        type=str,
        help="Path to the VCF file with SNPs. Format: filename.vcf. Can be compressed with bgzip or uncompressed.",
    )
    aln_group.add_argument(
        "--gtseq",
        default=None,
        type=str,
        help="Path to the GTSeq CSV file with SNPs. Format: filename.csv",
    )

    # Data Input and Preprocessing
    data_group = parser.add_argument_group(
        "Data Input and Preprocessing",
        description="Input files and input preprocessing options",
    )
    data_group.add_argument(
        "--sample_data",
        type=str,
        default=None,
        help="Tab-delimited file with 'sampleID', 'x', 'y'. Align SampleIDs with VCF",
    )
    data_group.add_argument(
        "--min_mac",
        type=validate_positive_int,
        default=2,
        help="Minimum minor allele count to retain SNPs. Default: 2.",
    )
    data_group.add_argument(
        "--max_SNPs",
        type=int,
        default=None,
        help="Max number of SNPs to randomly subset. Default: Use all SNPs.",
    )

    # Embedding settings.
    embed_group = parser.add_argument_group(
        "Embedding settings.",
        description="Settings for embedding the input features.",
    )
    embed_group.add_argument(
        "--embedding_type",
        type=validate_lower_str,
        default="none",
        help="Embedding to use with input SNP dataset. Supported options are: 'pca', 'polynomial', 'tsne', 'none' (no embedding). Default: 'none' (no embedding).",
    )
    embed_group.add_argument(
        "--n_components",
        default=None,
        help="Number of components to use with 'pca' or 'tsne' embeddings. If not specified, then 'n_components' will be optimized if using PCA, otherwise a value is required.'. Default: Search for optimal 'n_components.' parameter. Default: Search optimal components.",
    )
    embed_group.add_argument(
        "--embedding_sensitivity",
        type=validate_positive_float,
        default=1.0,
        help="Sensitivity setting for selecting optimal number of components with 'mca' and 'pca'. Set lower than 0 if you want fewer components, and higher than 0 for more components. Default: 1.0.",
    )
    embed_group.add_argument(
        "--tsne_perplexity",
        type=validate_positive_int,
        default=30,
        help="Perplexity setting if using T-SNE embedding. Default: 30.",
    )
    embed_group.add_argument(
        "--polynomial_degree",
        type=validate_positive_int,
        default=2,
        help="Polynomial degree to use with 'polynomial' embedding. WARNING: Setting this higher than 2 adds heavy computational overhead!!! Default: 2",
    )
    embed_group.add_argument(
        "--n_init",
        type=validate_positive_int,
        default=4,
        help="Number of initialization runs to use with Multi Dimensional Scaling embedding. Default: 4.",
    )

    # Model Configuration Arguments
    model_group = parser.add_argument_group(
        "Model Configuration", description="Model configuration arguments."
    )
    model_group.add_argument(
        "--nlayers",
        type=validate_positive_int,
        default=10,
        help="Number of hidden layers in the network. Default: 10.",
    )
    model_group.add_argument(
        "--width",
        type=validate_positive_int,
        default=256,
        help="Number of neurons (units) per layer. Default: 256.",
    )
    model_group.add_argument(
        "--dropout_prop",
        type=validate_positive_float,
        default=0.25,
        help="Dropout rate (0-1) to prevent overfitting. Default: 0.2.",
    )
    model_group.add_argument(
        "--load_best_params",
        type=str,
        default=None,
        help="Specify filename to load best paramseters from previous Optuna parameter search. Default: None (don't load best parameters).",
    )
    model_group.add_argument(
        "--use_gradient_boosting",
        action="store_true",
        help="Whether to use Gradient Boosting model instead of deep learning model. Default: False (use deep learning model).",
    )
    model_group.add_argument(
        "--dtype",
        type=validate_lower_str,
        default="float32",
        help="PyTorch data type to use. Supported options include: 'float32 and 'float64'. 'float64' is more accurate, but uses more memory and also is not supported with GPUs. Default: 'float32'.",
    )

    # Training Parameters
    training_group = parser.add_argument_group(
        "Training Parameters", description="Define model training parameters."
    )
    training_group.add_argument(
        "--batch_size",
        type=validate_positive_int,
        default=32,
        help="Training batch size. Default: 32.",
    )
    training_group.add_argument(
        "--max_epochs",
        type=validate_positive_int,
        default=5000,
        help="Max training epochs. Default: 5000.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=validate_positive_float,
        default=1e-3,
        help="Learning rate for optimizer. Default: 0.001.",
    )
    training_group.add_argument(
        "--l2_reg",
        type=validate_positive_float,
        default=0.0,
        help="L2 regularization weight. Default: 0 (none).",
    )
    training_group.add_argument(
        "--early_stop_patience",
        type=validate_positive_int,
        default=48,
        help="Epochs to wait before reducing learning rate after no improvement. Default: 100.",
    )
    training_group.add_argument(
        "--train_split",
        type=validate_split,
        default=0.9,
        help="Training data proportion (0-1). Default: 0.85.",
    )
    training_group.add_argument(
        "--val_split",
        type=validate_split,
        default=0.1,
        help="Validation data proportion (0-1). Default: 0.15.",
    )
    training_group.add_argument(
        "--do_bootstrap",
        action="store_true",
        default=False,
        help="Enable bootstrap replicates. Default: False.",
    )
    training_group.add_argument(
        "--nboots",
        type=validate_positive_int,
        default=50,
        help="Number of bootstrap replicates. Used if 'bootstrap' is True. Default: 50.",
    )
    training_group.add_argument(
        "--do_gridsearch",
        action="store_true",
        default=False,
        help="Perform grid search for parameter optimization. Default: False.",
    )
    training_group.add_argument(
        "--n_iter",
        type=validate_positive_int,
        default=100,
        help="Iterations for parameter optimization. Used with 'do_gridsearch'. Optuna recommends between 100-1000. Default: 100.",
    )
    training_group.add_argument(
        "--lr_scheduler_patience",
        default=17,
        type=validate_positive_int,
        help="Learning rate scheduler patience.",
    )
    training_group.add_argument(
        "--lr_scheduler_factor",
        type=validate_positive_float,
        default=0.5,
        help="Factor to reduce learning rate scheduler by.",
    )
    training_group.add_argument(
        "--factor",
        type=validate_positive_float,
        default=1.0,
        help="Factor to scale neural network widths by. Defaults to 1.0 (no width reduction)",
    )
    training_group.add_argument(
        "--grad_clip",
        action="store_true",
        help="If true, does gradient clipping to reduce overfitting.",
    )

    # Geographic Density Sampler Arguments
    geo_sampler_group = parser.add_argument_group("Geographic Density Sampler")
    geo_sampler_group.add_argument(
        "--use_weighted",
        type=validate_lower_str,
        default="none",
        help="Use inverse-weighted probability sampling to calculate sample weights based on sampling density; use the sample weights in the loss function, or both. Valid options include: 'sampler', 'loss', 'both', or 'none'. Default: 'none'.",
    )
    geo_sampler_group.add_argument(
        "--oversample_method",
        type=validate_lower_str,
        default="none",
        help="Synthetic oversampling/ undersampling method to use. Valid options include 'kmeans', 'optics', 'kerneldensity', or 'none'. Default: 'none' (do not use over-sampling).",
    )
    geo_sampler_group.add_argument(
        "--oversample_neighbors",
        type=validate_positive_int,
        default=5,
        help="Number of nearest neighbors to use with oversampling method. Default: 5.",
    )
    geo_sampler_group.add_argument(
        "--n_bins",
        type=validate_positive_int,
        default=8,
        help="Number of bins to use with synthetic resampling.",
    )
    geo_sampler_group.add_argument(
        "--use_kmeans",
        action="store_true",
        help="Use KMeans clustering in the Weighted Geographic Density Sampler. Default: False",
    )
    geo_sampler_group.add_argument(
        "--use_kde",
        action="store_true",
        default=True,
        help="Use Kernel Density Estimation in the Weighted Geographic Density Sampler. Default: True.",
    )
    geo_sampler_group.add_argument(
        "--w_power",
        type=validate_positive_float,
        default=1.0,
        help="Power for inverse density weighting. Set higher for more aggressive inverse weighting of sampling density. Default: 1.0",
    )
    geo_sampler_group.add_argument(
        "--max_clusters",
        type=validate_positive_int,
        default=10,
        help="Maximum number of clusters for KMeans when used with the geographic density sampler. Default: 10",
    )
    geo_sampler_group.add_argument(
        "--max_neighbors",
        type=validate_positive_int,
        default=50,
        help="Maximum number of nearest neighbors for adaptive bandwidth when doing geographic density sampling. Default: 50",
    )
    geo_sampler_group.add_argument(
        "--focus_regions",
        action=EvaluateAction,
        help="Provide geographic regions of interest to focus sampling density weights on. E.g., '[(lon_min1, lon_max1, lat_min1, lat_max1), ...]'.",
    )
    geo_sampler_group.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help="Whether to normalize density-based sample weights. Default: False (Do not normalize).",
    )

    outlier_detection_group = parser.add_argument_group(
        "Arguments for outlier detection based on IBD.",
        description="Parameters to adjust for the 'outlier_detection_group. This will perform outlier detection and remove significant outliers from the training and validation data.",
    )
    outlier_detection_group.add_argument(
        "--detect_outliers",
        action="store_true",
        help="Perform outlier detection to remove outliers.",
    )

    outlier_detection_group.add_argument(
        "--min_nn_dist",
        type=validate_positive_int,
        default=1000,
        help="Minimum required distance betewen nearest neighbors to consider outliers. This allows fine-tuning of outlier detection to exclude samples with geographic coordinates in very close proximity. Units are in meters. Default: 1000 (meters).",
    )

    outlier_detection_group.add_argument(
        "--scale_factor",
        type=validate_positive_int,
        default=100,
        help="Factor to scale geographic distance by. Helps with preventing errors with the Maximum Likelihood Estmiation when inferring the null gamma distribution to estimate p-values. Default: 100",
    )
    outlier_detection_group.add_argument(
        "--significance_level",
        type=validate_positive_float,
        default=0.05,
        help="Adjust the significance level (alpha) for P-values to determine significant outliers. Outliers <= 'significance_level' are removed. Must be in the range (0, 1). Default: 0.05.",
    )
    outlier_detection_group.add_argument(
        "--maxk",
        type=validate_positive_int,
        default=50,
        help="Maximum number of nearest neighbors (K) for outlier detection. K will be optimized between (2, maxk + 1). Default: 50.",
    )
    # Output and Miscellaneous Arguments
    output_group = parser.add_argument_group(
        "Output and Miscellaneous", description="Output and miscellaneous arguments."
    )
    output_group.add_argument(
        "--prefix",
        type=str,
        default="output",
        help="Output file prefix. Default: 'output'.",
    )
    output_group.add_argument(
        "--sqldb",
        type=str,
        default=None,
        help="SQLite3 database directory. Default: None (don't use database)",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Specify directory to store output files. Default: ./output.",
    )
    output_group.add_argument(
        "--seed",
        type=validate_seed,
        default=None,
        help="Random seed for reproducibility. Default: Random.",
    )
    output_group.add_argument(
        "--gpu_number",
        type=validate_gpu_number,
        default=None,
        help="GPU number for computation. If not specified, no GPU is used. Default: CPU usage (no GPU).",
    )
    output_group.add_argument(
        "--n_jobs",
        type=validate_n_jobs,
        default=-1,
        help="Number of CPU jobs to use. Default: -1 (use all CPUs).",
    )
    output_group.add_argument(
        "--verbose",
        type=validate_verbosity,
        default=1,
        help="Enable detailed logging. Verbosity level: 0 (non-verbose) to 3 (most verbose). Default: 1.",
    )

    plotting_group = parser.add_argument_group(
        "Plot Settings",
        description="Set plotting parameters to customize the visualizations.",
    )
    plotting_group.add_argument(
        "--show_plots",
        action="store_true",
        default=False,
        help="If True, then shows in-line plots. Useful if rendered in jupyter notebooks. Either way, the plots get saved to disk. Default: False (do not show in-line).",
    )
    plotting_group.add_argument(
        "--fontsize",
        type=validate_positive_int,
        default=24,
        help="Font size for plot axis labels, ticks, and titles. Default: 24.",
    )
    plotting_group.add_argument(
        "--filetype",
        type=validate_lower_str,
        default="png",
        help="File type to use for plotting. Valid options include any that 'matplotlib.pyplot.savefig' supports. Most common options include 'png' or 'pdf', but the following are supported: (eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp). Do not prepend a '.' character to the string. Default: 'png'.",
    )
    plotting_group.add_argument(
        "--plot_dpi",
        type=validate_positive_int,
        default=300,
        help="DPI to use for plots that are in raster format, such as 'png'. Default: 300.",
    )

    plotting_group.add_argument(
        "--shapefile_url",
        type=str,
        default="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        help="URL for shapefile used when plotting prediction error. This is a map of the continental USA, so if you need a different base map, you can supply your own URL here.",
    )
    plotting_group.add_argument(
        "--n_contour_levels",
        type=validate_positive_int,
        default=20,
        help="Number of contour levels to use in the plot that interpolates the prediction error on a spatial map (i.e., Kriging plot). Increase the for a more continuous distribution of contours, decrease it to visualize more discrete contour levels. Default: 20.",
    )
    plotting_group.add_argument(
        "--min_colorscale",
        type=int,
        default=0,
        help="Minimum colorbar value for the Kriging plot. Default: 0.",
    )
    plotting_group.add_argument(
        "--max_colorscale",
        type=validate_positive_int,
        default=300,
        help="Maximum value to use on the Kriging plot's colorbar. If your error distribution is higher than this value or you are getting uncontoured areas, increase this value. Default: 300.",
    )
    plotting_group.add_argument(
        "--sample_point_scale",
        type=validate_positive_int,
        default=2,
        help="Scale factor for sample point size on Kriging plot. If the sample points are too large or do not appear, decrease or increase this value, respectively. Default: 3.",
    )
    plotting_group.add_argument(
        "--bbox_buffer",
        type=validate_positive_float,
        default=0.1,
        help="Buffer to add to the sampling bounding box on map visualizations. Adjust to your liking. Default: 0.1.",
    )
    gb_group = parser.add_argument_group(
        title="Gradient Boosting Argument Group",
        description="Specify parameters for doing XGBoost predictions.",
    )
    gb_group.add_argument(
        "--gb_learning_rate",
        type=validate_positive_float,
        default=0.3,
        help="Step size shrinkage used in update to prevent overfitting. This parameter usually requires tuning. Default: 0.1.",
    )
    gb_group.add_argument(
        "--gb_n_estimators",
        type=validate_positive_int,
        default=100,
        help="Number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better perforamnce. Values must be in the range [1, inf). Default: 100.",
    )
    gb_group.add_argument(
        "--gb_subsample",
        type=validate_positive_float,
        default=1.0,
        help="Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration. Range: (0.0, 1.0]. Default: 1.0",
    )
    gb_group.add_argument(
        "--gb_colsample_bytree",
        type=validate_positive_float,
        default=1.0,
        help="colsample_bytree is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed. Range: (0.0, 1.0]. Default: 1.0",
    )

    gb_group.add_argument(
        "--gb_min_child_weight",
        default=1,
        help="Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be. Default: 1",
    )
    gb_group.add_argument(
        "--gb_max_delta_step",
        default=0,
        type=int,
        help="Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update. Default: 0 (no maximum delta step).",
    )
    gb_group.add_argument(
        "--gb_max_leaves",
        default=0,
        help="The maximum number of leaves allowed at a leaf node. Tuning this may have the effect of smoothing the model. Default: 0 (no maximum).",
    )
    gb_group.add_argument(
        "--gb_max_depth",
        default=6,
        help="Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.",
    )
    gb_group.add_argument(
        "--gb_reg_alpha",
        type=float,
        default=0,
        help="L1 regularization for Gradient Boosting model. Tuning this can mitigate overfitting. Range: [0, inf). Default: 0 (no L1 regularization).",
    )
    gb_group.add_argument(
        "--gb_reg_lambda",
        type=float,
        default=1,
        help="L2 regularization for Gradient Boosting model. Tuning this can mitigate overfitting. Range: [0, inf). Defaul: 0 (no L1 regularization).",
    )
    gb_group.add_argument(
        "--gb_gamma",
        type=float,
        default=0,
        help="Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. Range: [0, inf). Default: 0.0.",
    )
    gb_group.add_argument(
        "--gb_multi_strategy",
        type=str,
        default="one_output_per_tree",
        help="The strategy used for training multi-target models, including multi-target regression and multi-class classification. See Multiple Outputs for more information. Supported options: 'one_output_per_tree': Train a separate model for each target. 'multi_output_tree': Use multi-target trees. Default: 'one_target_per_node'.",
    )
    gb_group.add_argument(
        "--gb_objective",
        type=str,
        default="reg:squarederror",
        help="Loss objective to use for scoring with the gradient boosting model. Supported options: 'reg:squarederror', 'reg:quantileerror', 'reg:absoluteerror', 'reg:gamma', and 'reg:tweedie'. See XGBoost documentation for more information on each. Default: 'reg:squarederror'.",
    )
    gb_group.add_argument(
        "--gb_eval_metric",
        type=str,
        default="rmse",
        help="Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression. Supported options include: 'rmse' (root mean squared error), 'mae' (mean absolute error), 'mape' (mean absolute percentage error), 'gamma-nloglik' (negative log-likelihood for gamma regression), 'tweedie-nloglik' (negative log-likelihood for Tweedie regression). Default: 'rmse'.",
    )
    gb_group.add_argument(
        "--gb_early_stopping_rounds",
        type=validate_positive_int,
        default=10,
        help="Number of rounds to go before terminating training via early stopping criteria. Will revert the model to the best model. Default: 10.",
    )
    gb_group.add_argument(
        "--gb_use_lr_scheduler",
        action="store_true",
        help="Whether to use learning rate scheduler to gradually reduce the learning rate. Default: False (off).",
    )

    args = parser.parse_args()

    # Load and apply configuration file if provided
    validate_inputs(parser, args, test_mode=test_mode)
    validate_significance_levels(parser, args)
    validate_max_neighbors(parser, args)
    validate_embeddings(parser, args)
    validate_seed(args)
    validate_weighted_opts(parser, args)
    validate_colorscale(parser, args)
    validate_smote(parser, args)
    validate_gb_params(parser, args)
    validate_dtype(parser, args)

    return args


def validate_dtype(parser, args):
    if args.dtype not in ["float64", "float32"]:
        msg = f"'--dtype' argument must be either 'float64' or 'float32', but got: {args.dtype}"
        logger.error(msg)
        parser.error(msg)


def validate_gb_params(parser, args):
    if args.gb_objective not in [
        "reg:squarederror",
        "reg:squaredlogerror",
        "reg:absoluteerror",
    ]:
        msg = f"Invalid 'gb_objective' parameter provided. Supported options include: 'reg:squarederror', 'reg:squaredlogerror', 'reg:absoluteerror', but got: {args.gb_objective}"
        logger.error(msg)
        parser.error(msg)

    if args.gb_eval_metric not in [
        "rmse",
        "rmsle",
        "mae",
        "mape",
    ]:
        msg = f"Invalid parameter provided to 'gb_eval_metric'. Supported options include: 'rmse', 'rmsle', 'mae', 'mape', but got: {args.gb_eval_metric}."
        logger.error(msg)
        parser.error(msg)

    if args.gb_multi_strategy not in ["one_output_per_tree", "multi_output_tree"]:
        msg = f"Invalid parameter provided to 'gb_multi_strategy'. Supported options include 'one_output_per_tree' or 'multi_output_tree', but got: {args.gb_multi_strategy}."
        logger.error(msg)
        parser.error(msg)

    if args.gb_subsample > 1.0 or args.gb_subsample <= 0.0:
        msg = f"Invalid value provided for 'gb_subsample'. Values must be > 0.0 and <= 1.0, but got: {args.gb_subsample}"
        logger.error(msg)
        parser.error(msg)


def validate_weighted_opts(parser, args):
    if args.use_weighted not in ["sampler", "loss", "both", "none"]:
        msg = f"Invalid option passed to 'use_weighted': {args.use_weighted}"
        logger.error(msg)
        parser.error(msg)


def validate_colorscale(parser, args):
    if args.min_colorscale < 0:
        msg = f"'--min_colorscale' must be >= 0: {args.min_colorscale}"
        logger.error(msg)
        parser.error(msg)

    if args.max_colorscale <= args.min_colorscale:
        msg = f"'--max_colorscale must be > --min_colorscale', but got: {args.min_colorscale}, {args.max_colorscale}"
        logger.error(msg)
        parser.error(msg)


def validate_smote(parser, args):
    if args.oversample_method not in [
        "kmeans",
        "optics",
        "kerneldensity",
        "none",
    ]:
        msg = f"'--oversample_method' value must be one of 'kmeans', 'optics', 'kerneldensity', or 'none', but got: {args.oversample_method}'"
        logger.error(msg)
        parser.error(msg)


def validate_seed(args):
    if args.seed is not None and args.embedding_type == "polynomial":
        logger.warning(
            "'polynomial' embedding does not support a random seed, but a random seed was supplied to the 'seed' argument."
        )


def validate_embeddings(parser, args):
    if args.embedding_type.lower() not in [
        "pca",
        "tsne",
        "mds",
        "lle",
        "polynomial",
        "none",
        "kernelpca",
        "nmf",
        "mca",
    ]:
        msg = f"Invalid value supplied to '--embedding_type'. Supported options include: 'pca', 'tsne', 'mds', 'mca', 'polynomial', 'lle', 'kernelpca', 'nmf', or 'none', but got: {args.embedding_type}"
        logger.error(msg)
        parser.error(msg)

    if args.embedding_type.lower() == "polynomial":
        if args.polynomial_degree > 3:
            msg = f"'polynomial_degree' was set to {args.polynomial_degree}. Anything above 3 can add very large computational overhead!!! Use at your own risk!!!"
            warnings.warn(msg)
            logger.warning(msg)

    if args.n_components is not None:
        if args.n_components > 3 and args.embedding_type in ["tsne", "mds"]:
            msg = f"n_components must set to 2 or 3 to use 'tsne' and 'mds', but got: {args.n_components}"
            logger.error(msg)
            parser.error(msg)

    if args.n_components is None and args.embedding_type in ["tsne", "mds"]:
        msg = f"n_components must either be 2 or 3 if using 'tsne' or 'mds', but got NoneType."
        logger.error(msg)
        parser.error(msg)


def validate_max_neighbors(parser, args):
    if args.max_neighbors <= 1:
        logger.error(f"max_neighbors must be > 1: {args.max_neighbors}.")
        parser.error(f"max_neighbors must be > 1: {args.max_neighbors}.")

    if args.maxk <= 1:
        msg = f"max_neighbors must be > 1: {args.maxk}."
        logger.error(msg)
        parser.error(msg)


def validate_significance_levels(parser, args):
    if args.significance_level >= 1.0 or args.significance_level <= 0:
        msg = f"'significance_level' must be between 0 and 1: {args.significance_level}"
        logger.error(msg)
        parser.error(msg)

    if args.significance_level >= 0.5:
        logger.warning(
            f"'significance_level' was set to a high number: {args.significance_level}. Outliers are removed if the P-values are <= 'significance_level' (e.g., if P <= 0.05). Are you sure this is what you want?"
        )


def validate_inputs(parser, args, test_mode=False):
    if args.config:
        if not os.path.exists(args.config):
            parser.error(f"Configuration file not found: {args.config}")
        config = load_config(args.config)

        # Update default values based on the configuration file
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])

    if args.sample_data is None and not test_mode:
        logger.error("--sample_data argument is required.")
        parser.error("--sample_data argument is required.")

    if args.vcf is None and args.gtseq is None and not test_mode:
        logger.error("Either --vcf or --gtseq must be defined.")
        parser.error("Either --vcf or --gtseq must be defined.")

    if args.vcf is not None and args.gtseq is not None:
        logger.error("Only one of --vcf and --gtseq can be provided.")
        parser.error("Only one of --vcf and --gtseq can be provided.")
