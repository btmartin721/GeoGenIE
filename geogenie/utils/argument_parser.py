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
    aln_group = parser.add_argument_group(
        "Alignment Input (VCF File)", description="Input VCF file alignment to use."
    )
    aln_group.add_argument(
        "--vcf",
        default=None,
        type=str,
        help="Path to the VCF file with SNPs. Format: filename.vcf. Can be compressed with bgzip or uncompressed.",
    )

    # Data Input and Preprocessing
    data_group = parser.add_argument_group(
        "Data Input and Preprocessing",
        description="Input files and input preprocessing options",
    )
    data_group.add_argument(
        "--sample_data",
        default=None,
        type=str,
        help="Tab-delimited file with 'sampleID', 'x', 'y'. Align SampleIDs with VCF",
    )
    data_group.add_argument(
        "--known_sample_data",
        type=str,
        default=None,
        help="Same as sample_data. This is redundant, and may be removed in future versions, but is currently required if you want the recorded localities present on the CI plots. Default: None (no recorded localities shown)",
    )
    data_group.add_argument(
        "--min_mac",
        type=validate_positive_int,
        default=2,
        help="Minimum minor allele count (MAC) to retain SNPs. Default: 2.",
    )
    data_group.add_argument(
        "--max_SNPs",
        type=int,
        default=None,
        help="Max number of SNPs to randomly subset. Default: None (Use all SNPs).",
    )
    data_group.add_argument(
        "--prop_unknowns",
        type=validate_positive_float,
        default=0.1,
        help="Proportion of samples to randomly select for the 'unknown prediction' dataset if unknowns are not present in the '--sample_data' file. This setting only gets used if the '--sample_data' file contains no 'nan' values in place of coordinates. Default: 0.1",
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
        help="Embedding to use with input SNP dataset. Supported options are: 'pca', 'kernelpca', 'nmf', 'lle', 'mca', 'mds', 'polynomial', 'tsne', and 'none' (no embedding). Default: 'none' (no embedding).",
    )
    embed_group.add_argument(
        "--n_components",
        default=None,
        help="Number of components to use with 'pca' or 'tsne' embeddings. If not specified, then 'n_components' will be optimized if using PCA, otherwise a value is required.'. Default: Search for optimal 'n_components.' parameter. Default: None (Search optimal components).",
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
        help="Number of hidden layers in the network. Increase for underfitting. Default: 10.",
    )
    model_group.add_argument(
        "--width",
        type=validate_positive_int,
        default=256,
        help="Number of neurons per layer. If `--factor` is less than 1.0, then `width` will be reduced with each successive hidden layer, and `width` would represent the initial width. Default: 256.",
    )
    model_group.add_argument(
        "--dropout_prop",
        type=validate_positive_float,
        default=0.25,
        help="Dropout rate (0-1) to prevent overfitting. Default: 0.2.",
    )
    model_group.add_argument(
        "--criterion",
        type=validate_lower_str,
        default="rmse",
        help=f"Model loss criterion to use. Valid options include: 'rmse', 'huber', and 'drms'. RMSE is Root Mean Squared Error (i.e., euclidean distance) and 'drms' is Distance Root Mean Square. 'huber' is Huber loss. All criteria are weighted if using the '--use_weighted' argument with 'loss', 'sampler', or 'both'.",
    )
    model_group.add_argument(
        "--load_best_params",
        type=str,
        default=None,
        help="Specify filename to load best paramseters from previous Optuna parameter search Should be a .json file. Can be found in '<output_dir>/optimize/<prefix>_best_params.json'. Default: None (don't load best parameters).",
    )
    model_group.add_argument(
        "--use_gradient_boosting",
        action="store_true",
        help="Whether to use Gradient Boosting model instead of deep learning model. This method is deprecated and may be removed in a future version. Non-functional. Default: False (use deep learning model).",
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
        help="`--max_epochs`: Maximum training epochs. An early stopping mechanism is implemented, so it is advisable to set this to a very high number and let the early stopping mechanism determine when to stop training. Default: 5000.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=validate_positive_float,
        default=1e-3,
        help="`--learning_rate`: Learning rate for optimizer. Subject to a learning rate scheduler that reduces the learning rate with no improvement after `lr_scheduler_patience` epochs. Default: 0.001.",
    )
    training_group.add_argument(
        "--l2_reg",
        type=validate_positive_float,
        default=0.0,
        help="`--l2_reg`: L2 regularization weight. Can help to reduce overfitting. Default: 0.0 (no regularization).",
    )
    training_group.add_argument(
        "--early_stop_patience",
        type=validate_positive_int,
        default=48,
        help="Epochs to wait after no improvement before activating early stopping mechanism. Default: 100.",
    )
    training_group.add_argument(
        "--train_split",
        type=validate_split,
        default=0.8,
        help="Training data proportion. `--val_split` + `--train_split` must sum to 1.0. NOTE: `--train_split` will be further reduced by `--train_split - `--val_split` to create a 'test sample' hold-out subset. Default: 0.8",
    )
    training_group.add_argument(
        "--val_split",
        type=validate_split,
        default=0.1,
        help="--val_split`: Validation data proportion. `--val_split` + `--train_split` must sum to 1.0. Default: 0.2.",
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
        default=100,
        help="`--nboots`: Number of bootstrap replicates. Has no effect if `--do_bootstrap` is disabled. Default: 100.",
    )
    training_group.add_argument(
        "--do_gridsearch",
        action="store_true",
        default=False,
        help="Perform Optuna parameter search to optimize model parameters. Default: False.",
    )
    training_group.add_argument(
        "--n_iter",
        type=validate_positive_int,
        default=100,
        help="Iterations for parameter optimization. Used with 'do_gridsearch'. Optuna recommends between 100-1000. Default: 100.",
    )
    training_group.add_argument(
        "--lr_scheduler_patience",
        default=16,
        type=validate_positive_int,
        help="Learning rate scheduler patience.",
    )
    training_group.add_argument(
        "--lr_scheduler_factor",
        type=validate_positive_float,
        default=0.5,
        help="Factor to reduce learning rate when learning rate scheduler is triggered. Default: 0.5.",
    )
    training_group.add_argument(
        "--factor",
        type=validate_positive_float,
        default=1.0,
        help="Factor to scale neural network widths with each successive hidden layer. Default: 1.0 (no width reduction).",
    )
    training_group.add_argument(
        "--grad_clip",
        action="store_true",
        help="Enable gradient clipping, which can reduce the effect of explosive gradients. Default: False.",
    )

    # Geographic Density Sampler Arguments
    geo_sampler_group = parser.add_argument_group("Geographic Density Sampler")
    geo_sampler_group.add_argument(
        "--use_weighted",
        type=validate_lower_str,
        default="none",
        help="Use inverse-weighted probability sampling to calculate sample weights based on sampling density; use the sample weights in the loss function, or both. Valid options include: 'loss' or 'none'. Default: 'none'.",
    )
    geo_sampler_group.add_argument(
        "--oversample_method",
        type=validate_lower_str,
        default="none",
        help="Synthetic oversampling/ undersampling method to use. Valid options include 'kmeans' or 'none'. Default: 'none' (no oversampling).",
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
        help="Number of KMeans bins (i.e., K clusters) to use with synthetic resampling.",
    )
    geo_sampler_group.add_argument(
        "--use_kmeans",
        action="store_true",
        default=False,
        help="Use KMeans clustering to calculate sample weights. Default: False",
    )
    geo_sampler_group.add_argument(
        "--use_kde",
        action="store_true",
        default=False,
        help="Use Kernel Density Estimation (KDE) in to calculate sample weights. Default: False.",
    )
    geo_sampler_group.add_argument(
        "--use_dbscan",
        action="store_true",
        default=False,
        help="Use DBSCAN clustering to calculate sample weights. This method is still experimental. Use with caution. Default: False.",
    )
    geo_sampler_group.add_argument(
        "--w_power",
        type=validate_positive_float,
        default=1.0,
        help="Exponential Power for inverse density weighting. Set this to a larger value if the sampling densities are not differentiated enough. Default: 1.0.",
    )
    geo_sampler_group.add_argument(
        "--max_clusters",
        type=validate_positive_int,
        default=10,
        help="Maximum number of clusters for KMeans when used with the to calculate sample weights. Default: 10",
    )
    geo_sampler_group.add_argument(
        "--max_neighbors",
        type=validate_positive_int,
        default=50,
        help="Maximum number of nearest neighbors for adaptive bandwidth when doing geographic density sampling. Argument is deprecated nad will be removed in a future version. Default: 50",
    )
    geo_sampler_group.add_argument(
        "--focus_regions",
        action=EvaluateAction,
        help="Specify geographic regions of interest using minimum and maximum longitude and latitude coordinates for sampling density weights. Should be in the format: `[(lon_min1, lon_max1, lat_min1, lat_max1), (<region2>), (...), (<regionN>)]`. Multiple regions can be specified.",
    )
    geo_sampler_group.add_argument(
        "--normalize_sample_weights",
        action="store_true",
        help="Whether to normalize density-based sample weights from 0 to 1. Default: False (no normalization).",
    )

    outlier_detection_group = parser.add_argument_group(
        "Arguments for outlier detection based on IBD.",
        description="Parameters to adjust for the 'outlier_detection_group. This will perform outlier detection and remove significant outliers from the data prior to training.",
    )
    outlier_detection_group.add_argument(
        "--detect_outliers",
        action="store_true",
        default=False,
        help="Enable outlier detection to remove geographic and/ or genetic outliers. Default: False.",
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
        help="Output file prefix. Used for all output files and plots. Default: 'output'.",
    )
    output_group.add_argument(
        "--sqldb",
        type=str,
        default=None,
        help="SQLite3 database directory to use with Optuna parameeter optimization. Allows parameter tuning to be resumed. Default: None (no database, with Optuna non-resumeable).",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to store output files and plots. Default: './output'.",
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
    output_group.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Toggles on debug mode, which adds many debug statements to the logs. Default: False (no debug mode)",
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
        "--remove_splines",
        action="store_true",
        default=False,
        help="Remove axis splines from map plots. Defaults to False (don't remoe splines).",
    )

    plotting_group.add_argument(
        "--shapefile",
        type=str,
        default="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_500k.zip",
        help=f"URL or file path for shapefile used when plotting prediction error. The default is a map of the continental USA, so if you need a different base map, you can supply your own URL or zipped shapefile here. Note that if '--basemap_fips_code is not provided, then the map will be zoomed to the bounding box of the samples. plus '--bbox_buffer. Default: Continental USA basemap, downloaded from census.gov URL.",
    )
    plotting_group.add_argument(
        "--basemap_fips",
        type=str,
        default=None,
        help="FIPS code for basemap. If provided, the bsae map will be of the US state for the provided FIPS code. If no FIPS code is provided, then the base map will be zoomed to the sampling bounding box, plus '--bbox_buffer'. Default: Do not use FIPS code.",
    )
    plotting_group.add_argument(
        "--highlight_basemap_counties",
        type=str,
        default=None,
        help="Comma-separated string of county names to use with the base map. Provided counties will be highlighted in gray on the base map. If not provided, then no counties are highlighted. Default: Do not highlight any counties.",
    )

    plotting_group.add_argument(
        "--samples_to_plot",
        type=str,
        default=None,
        help="Comma-separated string of sampleIDs to plot contour predictions for. Provided sampleIDs will be plotted over a base map with contours signifying 90, 70, and 50 percent of the bootstrap density (i.e., density of all predicted localities from the bootstrap replicates). If not provided, then all samples are plotted in separate plots. If set to an integer, then it will select '--samples_to_plot' random samples to plot. Default: None (Plot all samples).",
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
    args = parser.parse_args()

    if args.use_gradient_boosting:
        msg = "Gradient Boosting is currently not supported. Please use the deep learning model. This feature may be removed in a future version."
        logger.error(msg)
        parser.error(msg)

    # Load and apply configuration file if provided
    validate_inputs(parser, args, test_mode=test_mode)
    validate_significance_levels(parser, args)
    validate_max_neighbors(parser, args)
    validate_embeddings(parser, args)
    validate_seed(args.seed)
    args = validate_weighted_opts(parser, args)
    validate_colorscale(parser, args)
    validate_smote(parser, args)
    # validate_gb_params(parser, args)
    validate_dtype(parser, args)
    args.samples_to_plot = validate_str2list(args.samples_to_plot)
    args.highlight_basemap_counties = validate_str2list(args.highlight_basemap_counties)

    if args.debug:
        args.verbose = 2

    return args


def validate_str2list(arg):
    if arg is not None:
        if not arg.isdigit():
            if isinstance(arg, str):
                s = arg.strip()
                l = s.split(",")
                l = [x.strip() for x in l]
                return l
    return arg


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
    if args.use_weighted not in {"loss", "none"}:
        msg = f"Invalid option passed to 'use_weighted': {args.use_weighted}"
        logger.error(msg)
        parser.error(msg)

    return args


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
        "kerneldensity",
        "none",
    ]:
        msg = f"'--oversample_method' value must be one of 'kmeans', 'kerneldensity', or 'none', but got: {args.oversample_method}'"
        logger.error(msg)
        parser.error(msg)


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

    if args.vcf is None:
        logger.error("--vcf argument is required.")
        parser.error("--vcf argument is required.")
