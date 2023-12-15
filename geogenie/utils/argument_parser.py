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


def validate_model_type(value):
    """Validate the model type."""
    model_types = ["mlp", "gcn", "transformer"]
    if value not in model_types:
        raise argparse.ArgumentTypeError(f"Invalid model_type argument: {value}")
    return value


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


def validate_model_type(value):
    if value is None:
        raise TypeError("'model_type' cannot be NoneType.")
    try:
        model_type = str(value)
        if model_type not in ["mlp", "gcn", "transformer"]:
            raise ValueError(f"Invalid model_type argument provided: {model_type}")
    except (TypeError, ValueError):
        raise TypeError(
            f"Could not convert 'model_type' to a string: {type(model_type)}"
        )
    return model_type


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
    return gpu_number


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


def setup_parser():
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
        type=str,
        default="pca",
        help="Embedding to use with input SNP dataset. Supported options are: 'pca', 'polynomial', 'tsne', 'none' (no embedding). Default: 'pca'.",
    )
    embed_group.add_argument(
        "--n_components",
        default=None,
        help="Number of components to use with 'pca' or 'tsne' embeddings. If not specified, then 'n_components' will be optimized if using PCA, otherwise a value is required.'. Default: Search for optimal 'n_components.' parameter. Default: Search optimal components.",
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
        "--model_type",
        type=validate_model_type,
        default="mlp",
        help="Specify model type. Supported options: 'mlp', 'gcn', 'transformer'.",
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
        default=0.2,
        help="Dropout rate (0-1) to prevent overfitting. Default: 0.2.",
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
        "--patience",
        type=validate_positive_int,
        default=48,
        help="Epochs to wait before reducing learning rate after no improvement. Default: 100.",
    )
    training_group.add_argument(
        "--train_split",
        type=validate_split,
        default=0.85,
        help="Training data proportion (0-1). Default: 0.85.",
    )
    training_group.add_argument(
        "--val_split",
        type=validate_split,
        default=0.15,
        help="Validation data proportion (0-1). Default: 0.15.",
    )
    training_group.add_argument(
        "--class_weights",
        action="store_true",
        default=False,
        help="Apply class weights to account for imbalanced geographic sampling density'. Default: False",
    )
    training_group.add_argument(
        "--bootstrap",
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
        "--alpha",
        default=0.5,
        type=validate_positive_float,
        help="Factor to scale havesine loss by.",
    )
    training_group.add_argument(
        "--beta",
        default=0.5,
        type=validate_positive_float,
        help="Factor to scale mean and StdDev by in loss function.",
    )
    training_group.add_argument(
        "--gamma",
        default=0.5,
        type=validate_positive_float,
        help="Factor to scale r-squared by in loss function.",
    )
    training_group.add_argument(
        "--lr_scheduler_patience",
        default=8,
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
        default=0.5,
        help="Factor to scale neural network widths by.",
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
        type=str,
        help="Use inverse-weighted probability sampling to calculate sample weights based on sampling density; use the sample weights in the loss function, or both. Valid options include: 'sampler', 'loss', 'both', or 'none'. Default: 'none'.",
    )
    geo_sampler_group.add_argument(
        "--use_synthetic_oversampling",
        action="store_true",
        help="Use synthetic oversampling of low-density regions to get better predictions.",
    )
    geo_sampler_group.add_argument(
        "--use_kmeans",
        action="store_true",
        help="Use KMeans clustering in the Weighted Geographic Density Sampler. Default: False",
    )
    geo_sampler_group.add_argument(
        "--use_kde",
        action="store_false",
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

    transformer_group = parser.add_argument_group(
        "Transformer-specific model parameters.",
        description="Parameters to adjust for the 'transformer' model_type only.",
    )
    transformer_group.add_argument(
        "--embedding_dim",
        type=validate_positive_int,
        default=256,
        help="The size of the embedding vectors. It defines the dimensionality of the input and output tokens in the model. Higher dimensions can capture more information but increase computational complexity.",
    )
    transformer_group.add_argument(
        "--nhead",
        type=validate_positive_int,
        default=8,
        help="In multi-head attention, this parameter defines the number of parallel attention heads used. More heads allow the model to simultaneously attend to information from different representation subspaces, potentially capturing a wider range of dependencies.",
    )
    transformer_group.add_argument(
        "--dim_feedforward",
        type=validate_positive_int,
        default=1024,
        help="The size of the inner feedforward networks within each transformer layer. Adjusting this can impact the model’s ability to process information within each layer.",
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
        default="./database",
        help="SQLite3 database directory. Default: ./database.",
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
    output_group.add_argument(
        "--show_plots",
        action="store_true",
        default=False,
        help="If True, then shows in-line plots. Default: False (do not show in-line). Either way, the plots also get saved to disk.",
    )
    output_group.add_argument(
        "--fontsize",
        type=validate_positive_int,
        default=18,
        help="Font size for plot axis labels and title. Default: 18.",
    )

    output_group.add_argument(
        "--shapefile_url",
        type=str,
        default="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        help="URL for shapefile used for plotting prediction error. This is a map of the continental USA, so if you need a different base map, you can supply your own URL here.",
    )

    args = parser.parse_args()

    # Load and apply configuration file if provided
    if args.config:
        if not os.path.exists(args.config):
            parser.error(f"Configuration file not found: {args.config}")
        config = load_config(args.config)

        # Update default values based on the configuration file
        for arg in vars(args):
            if arg in config:
                setattr(args, arg, config[arg])

    if args.sample_data is None:
        logger.error("--sample_data argument is required.")
        parser.error("--sample_data argument is required.")

    if args.vcf is None and args.gtseq is None:
        logger.error("Either --vcf or --gtseq must be defined.")
        parser.error("Either --vcf or --gtseq must be defined.")

    if args.vcf is not None and args.gtseq is not None:
        logger.error("Only one of --vcf and --gtseq can be provided.")
        parser.error("Only one of --vcf and --gtseq can be provided.")

    if args.significance_level >= 1.0 or args.significance_level <= 0:
        msg = f"'significance_level' must be between 0 and 1: {args.significance_level}"
        logger.error(msg)
        parser.error(msg)

    if args.significance_level >= 0.5:
        logger.warning(
            f"'significance_level' was set to a high number: {args.significance_level}. Outliers are removed if the P-values are <= 'significance_level' (e.g., if P <= 0.05). Are you sure this is what you want?"
        )

    if args.max_neighbors <= 1:
        logger.error(f"max_neighbors must be > 1: {args.max_neighbors}.")
        parser.error(f"max_neighbors must be > 1: {args.max_neighbors}.")

    if args.maxk <= 1:
        msg = f"max_neighbors must be > 1: {args.maxk}."
        logger.error(msg)
        parser.error(msg)

    if args.embedding_type.lower() not in [
        "pca",
        "tsne",
        "mds",
        "lle",
        "polynomial",
        "none",
    ]:
        msg = f"Invalid value supplied to '--embedding_type'. Supported options include: 'pca', 'tsne', 'mds', 'polynomial', 'lle', 'none', but got: {args.embedding_type}"
        logger.error(msg)
        parser.error(msg)

    if args.embedding_type.lower() == "polynomial":
        if args.polynomial_degree > 3:
            msg = f"'polynomial_degree' was set to {args.polynomial_degree}. Anything above 3 can add very large computational overhead!!! Use at your own risk!!!"
            warnings.warn(msg)
            logger.warning(msg)

    if args.embedding_type.lower() == "none":
        msg = "'--embedding_type' was set to 'none', which means no embedding will be performed. Did you intend to do this?"
        logger.wanring(msg)
        warnings.warn(msg)

    if args.seed is not None and args.embedding_type == "polynomial":
        logger.warning(
            "'polynomial' embedding does not support a random seed, but a random seed was supplied to the 'seed' argument."
        )

    if args.n_components is not None:
        if args.n_components > 3 and args.embedding_type in ["tsne", "mds"]:
            msg = f"n_components must set to 2 or 3 to use 'tsne' and 'mds', but got: {args.n_components}"
            logger.error(msg)
            parser.error(msg)

    if args.n_components is None and args.embedding_type in ["tsne", "mds"]:
        msg = f"n_components must either be 2 or 3 if using 'tsne' or 'mds', but got NoneType."
        logger.error(msg)
        parser.error(msg)

    if args.use_weighted not in ["sampler", "loss", "both", "none"]:
        msg = f"Invalid option passed to 'use_weighted': {args.use_weighted}"
        logger.error(msg)
        parser.error(msg)

    return args
