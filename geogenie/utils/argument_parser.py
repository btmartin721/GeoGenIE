import argparse
import os

import yaml
from torch.cuda import is_available

from geogenie.utils.exceptions import GPUUnavailableError, ResourceAllocationError


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


class ProcessPopMapAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values if values is not None else False)


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
                raise ValueError("'gpu_number' must be > 0")
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


def validate_n_iter(value):
    try:
        n_iter = int(value)
        if n_iter <= 1:
            raise ValueError(f"'n_iter' value must be > 1: {n_iter}")
    except ValueError:
        raise ValueError(f"{n_iter} value must be > 1.")
    return n_iter


def validate_dropout(value):
    try:
        dropout = float(value)
        if dropout >= 1.0 or dropout < 0.0:
            raise ValueError(f"'dropout_prop' must be >= 0 and < 1: {dropout}")
    except ValueError:
        raise ValueError(f"'dropout value must be >= 0 and < 1: {dropout}")
    return dropout


def validate_epochs(value):
    try:
        epochs = int(value)
        if epochs <= 0:
            raise ValueError(f"'max_epochs' must be > 0: {epochs}")
    except ValueError:
        raise ValueError(f"'max_epochs' value must be > 0: {epochs}")
    return epochs


def validate_lr(value):
    try:
        lr = float(value)
        if lr <= 0.0 or lr > 1.0:
            raise ValueError(f"'learning_rate' must be > 0 and < 1.0: {lr}")
    except ValueError:
        raise ValueError(f"'learning_rate' must be > 0 and < 1.0: {lr}")
    return lr


def validate_l2(value):
    try:
        l2 = float(value)
        if l2 < 0.0 or l2 >= 1.0:
            raise ValueError(f"'l2_reg' must be >= 0 and < 1.0: {l2}")
    except ValueError:
        raise ValueError(f"'l2_reg' must be >= 0 and < 1.0: {l2}")
    return l2


def validate_transformer(value):
    try:
        val = int(value)
        if val <= 0:
            raise ValueError(
                f"Transformer parameter settings must integers > 0, but got: {value}"
            )
    except ValueError:
        raise ValueError(f"Transformer parameter settings must integers > 0: {value}")
    return val


def validate_patience(value):
    try:
        pat = int(value)
        if pat <= 0:
            raise ValueError(f"'patience' must be > 0: {pat}")
    except ValueError:
        raise ValueError(f"'patience' value must be > 0: {pat}")
    return pat


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


def validate_nboot(value):
    try:
        nboot = int(value)
        if nboot < 2:
            raise ValueError(f"'nboots' must be > 1: {nboot}")
    except ValueError:
        raise ValueError(f"'nboots must be > 1: {nboot}")
    return nboot


def validate_verbosity(value):
    try:
        verb = int(value)
        if verb < 0 or verb > 2:
            raise ValueError(f"'verbose' must >= 0 and <= 2: {verb}")
    except ValueError:
        raise ValueError(f"'verbose' must >= 0 and <= 2: {value}")
    return verb


def validate_outlier_scaler(value):
    try:
        outlier = float(value)
        if outlier <= 0.0 or outlier >= 1.0:
            raise ValueError(
                f"'outlier_detection_scaler' must be > 0 and < 1, but got: {outlier}"
            )
    except ValueError:
        raise ValueError(
            f"'outlier_detection_scaler' must be > 0 and < 1, but got: {value}"
        )


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
        help="Path to the VCF file with SNPs. Format: filename.vcf.",
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
        "--popmap",
        default=None,
        nargs="?",
        action=ProcessPopMapAction,
        help="Tab-delimited file with 'sampleID' and 'populationID'. Required if 'class_weights' is True.",
    )
    data_group.add_argument(
        "--min_mac",
        type=int,
        default=2,
        help="Minimum minor allele count to retain SNPs. Default: 2.",
    )
    data_group.add_argument(
        "--max_SNPs",
        type=int,
        default=None,
        help="Max number of SNPs to randomly subset. Default: Use all SNPs.",
    )
    data_group.add_argument(
        "--impute_missing",
        action="store_false",
        help="If True, imputes missing values from binomial distribution. Default: True",
    )
    data_group.add_argument(
        "--outlier_detection_scaler",
        default=0.3,
        help="Scaler to remove outliers from training/ validation data. Adjust if too many or too few samples are getting removed. Must be between 0 and 1.",
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
        type=int,
        default=10,
        help="Hidden layers in the network. Default: 10.",
    )
    model_group.add_argument(
        "--width", type=int, default=256, help="Neurons per layer. Default: 256."
    )
    model_group.add_argument(
        "--dropout_prop",
        type=validate_dropout,
        default=0.2,
        help="Dropout rate (0-1) to prevent overfitting. Default: 0.2.",
    )

    # Training Parameters
    training_group = parser.add_argument_group(
        "Training Parameters", description="Define model training parameters."
    )
    training_group.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size. Default: 32."
    )
    training_group.add_argument(
        "--max_epochs",
        type=validate_epochs,
        default=5000,
        help="Max training epochs. Default: 5000.",
    )
    training_group.add_argument(
        "--learning_rate",
        type=validate_lr,
        default=1e-3,
        help="Learning rate for optimizer. Default: 0.001.",
    )
    training_group.add_argument(
        "--l2_reg",
        type=validate_l2,
        default=0.0,
        help="L2 regularization weight. Default: 0 (none).",
    )
    training_group.add_argument(
        "--patience",
        type=validate_patience,
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
        help="Apply class weights for imbalanced datasets. Requires 'popmap'.",
    )
    training_group.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Enable bootstrap replicates. Default: False.",
    )
    training_group.add_argument(
        "--nboots",
        type=validate_nboot,
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
        type=int,
        default=100,
        help="Iterations for parameter optimization. Used with 'do_gridsearch'. Optuna recommends between 100-1000. Default: 100.",
    )
    transformer_group = parser.add_argument_group(
        "Transformer-specific model parameters.",
        description="Parameters to adjust for the 'transformer' model_type only.",
    )
    transformer_group.add_argument(
        "--embedding_dim",
        type=validate_transformer,
        default=256,
        help="The size of the embedding vectors. It defines the dimensionality of the input and output tokens in the model. Higher dimensions can capture more information but increase computational complexity.",
    )
    transformer_group.add_argument(
        "--nhead",
        type=validate_transformer,
        default=8,
        help="In multi-head attention, this parameter defines the number of parallel attention heads used. More heads allow the model to simultaneously attend to information from different representation subspaces, potentially capturing a wider range of dependencies.",
    )
    transformer_group.add_argument(
        "--dim_feedforward",
        type=validate_transformer,
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
        "--show_progress_bar",
        action="store_true",
        default=False,
        help="Show a tqdm progress bar during optimization. Default: False.",
    )
    output_group.add_argument(
        "--verbose",
        type=validate_verbosity,
        default=1,
        help="Enable detailed logging. Verbosity level: 0 (non-verbose) to 2 (most verbose). Default: 1.",
    )
    output_group.add_argument(
        "--show_plots",
        action="store_true",
        default=False,
        help="If True, then shows in-line plots. Default: False (do not show).",
    )
    output_group.add_argument(
        "--fontsize",
        type=int,
        default=18,
        help="Font size for plot axis labels and title. Default: 18.",
    )

    output_group.add_argument(
        "--shapefile_url",
        type=str,
        default="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        help="URL for shapefile used for plotting prediction error.",
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

    # Post-parsing validation
    if args.class_weights and not args.popmap:
        parser.error("--class_weights requires --popmap to be specified.")

    if args.sample_data is None:
        parser.error("--sample_data argument is required.")

    if args.vcf is None and args.gtseq is None:
        parser.error("Either --vcf or --gtseq must be defined.")

    if args.vcf is not None and args.gtseq is not None:
        parser.error("Only one of --vcf and --gtseq can be provided.")

    return args
