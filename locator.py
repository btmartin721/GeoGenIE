# estimating sample locations from genotype matrices
import argparse
import copy
import csv
import json
import logging
import os
import subprocess
import sys
import time
from functools import wraps
from pathlib import Path

import allel
import numpy as np
import pandas as pd
import tensorflow as tf
import zarr
from scipy import spatial
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from tensorflow.keras import backend as K
from tqdm import tqdm

from geogenie.plotting.plotting import PlotGenIE
from geogenie.utils.scorers import calculate_rmse, haversine_distances_agg, kstest

Path("locator/plots").mkdir(parents=True, exist_ok=True)
Path("locator/plots/plots").mkdir(parents=True, exist_ok=True)
Path("locator/test").mkdir(parents=True, exist_ok=True)
Path("locator/shapefile").mkdir(parents=True, exist_ok=True)
Path("locator/plots/shapefile").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

plotting = PlotGenIE(
    "cpu", "locator", "original_locator_2", fontsize=24, filetype="pdf"
)

execution_times = []


def timer(func):
    """
    Decorator that measures and stores the execution time of a function.

    Args:
        func (Callable): The function to be wrapped by the timer.

    Returns:
        Callable: The wrapped function with timing functionality.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append((func.__name__, execution_time))
        return result

    return wrapper


def save_execution_times(filename):
    """
    Appends the execution times to a CSV file. If the file doesn't exist, it creates one.

    Args:
        filename (str): The name of the file where data will be saved.
    """
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        # Check if the file is empty to decide whether to write headers
        f.seek(0, 2)  # Move to the end of the file
        if f.tell() == 0:  # Check if file is empty
            # Write headers only if file is empty
            writer.writerow(["Function Name", "Execution Time"])
        writer.writerows(execution_times)


parser = argparse.ArgumentParser()
parser.add_argument("--vcf", help="VCF with SNPs for all samples.")
parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
parser.add_argument(
    "--matrix",
    help="tab-delimited matrix of minor allele counts with first column named 'sampleID'.\
        E.g., \
        \
        sampleID\tsite1\tsite2\t...\n \
        msp1\t0\t1\t...\n \
        msp2\t2\t0\t...\n ",
)
parser.add_argument(
    "--sample_data",
    help="tab-delimited text file with columns\
        'sampleID \t x \t y'.\
        SampleIDs must exactly match those in the \
        VCF. X and Y values for \
        samples without known locations should \
        be NA.",
)
parser.add_argument(
    "--train_split",
    default=0.9,
    type=float,
    help="0-1, proportion of samples to use for training. \
        default: 0.9 ",
)
parser.add_argument(
    "--windows",
    default=False,
    action="store_true",
    help="Run windowed analysis over a single chromosome (requires zarr input).",
)
parser.add_argument("--window_start", default=0, help="default: 0")
parser.add_argument("--window_stop", default=None, help="default: max snp position")
parser.add_argument("--window_size", default=5e5, help="default: 500000")
parser.add_argument(
    "--bootstrap",
    default=False,
    action="store_true",
    help="Run bootstrap replicates by retraining on bootstrapped data.",
)
parser.add_argument(
    "--jacknife",
    default=False,
    action="store_true",
    help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic -- use the bootstrap \
                    option or run windowed analyses for final results.",
)
parser.add_argument(
    "--jacknife_prop",
    default=0.05,
    type=float,
    help="proportion of SNPs to remove for jacknife resampling.\
                    default: 0.05",
)
parser.add_argument(
    "--nboots",
    default=50,
    type=int,
    help="number of bootstrap replicates to run.\
                    default: 50",
)
parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
parser.add_argument("--max_epochs", default=5000, type=int, help="default: 5000")
parser.add_argument(
    "--patience",
    type=int,
    default=100,
    help="n epochs to run the optimizer after last \
                          improvement in validation loss. \
                          default: 100",
)
parser.add_argument(
    "--min_mac",
    default=2,
    type=int,
    help="minimum minor allele count.\
                          default: 2.",
)
parser.add_argument(
    "--max_SNPs",
    default=None,
    type=int,
    help="randomly select max_SNPs variants to use in the analysis \
                    default: None.",
)
parser.add_argument(
    "--impute_missing",
    default=False,
    action="store_true",
    help="default: True (if False, all alleles at missing sites are ancestral)",
)
parser.add_argument(
    "--dropout_prop",
    default=0.25,
    type=float,
    help="proportion of weights to zero at the dropout layer. \
                           default: 0.25",
)
parser.add_argument(
    "--nlayers",
    default=10,
    type=int,
    help="number of layers in the network. \
                        default: 10",
)
parser.add_argument(
    "--width",
    default=256,
    type=int,
    help="number of units per layer in the network\
                    default:256",
)
parser.add_argument("--out", help="file name stem for output")
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed for train/test splits and SNP subsetting.",
)
parser.add_argument("--gpu_number", default=None, type=str)
parser.add_argument(
    "--plot_history",
    default=True,
    type=bool,
    help="plot training history? \
                    default: True",
)
parser.add_argument(
    "--gnuplot",
    default=False,
    action="store_true",
    help="print acii plot of training history to stdout? \
                    default: False",
)
parser.add_argument(
    "--keep_weights",
    default=False,
    action="store_true",
    help="keep model weights after training? \
                    default: False.",
)
parser.add_argument(
    "--load_params",
    default=None,
    type=str,
    help="Path to a _params.json file to load parameters from a previous run.\
                          Parameters from the json file will supersede all parameters provided \
                          via command line.",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
)
args = parser.parse_args()

# set seed and gpu
if args.seed is not None:
    np.random.seed(args.seed)
if args.gpu_number is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

# load old run parameters
if args.load_params is not None:
    with open(args.predict_from_weights + "_params", "r") as f:
        args.__dict__ = json.load(f)
    f.close()

# store run params
with open("locator/" + args.out + "_params.json", "w") as f:
    json.dump(args.__dict__, f, indent=2)
f.close()


def load_genotypes():
    if args.zarr is not None:
        print("reading zarr")
        callset = zarr.open_group(args.zarr, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        positions = callset["variants/POS"]
    elif args.vcf is not None:
        print("reading VCF")
        vcf = allel.read_vcf(args.vcf, log=sys.stderr)
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]
    elif args.matrix is not None:
        gmat = pd.read_csv(args.matrix, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        gmat = np.array(gmat, dtype="int8")
        for i in range(
            gmat.shape[0]
        ):  # kludge to get haplotypes for reading in to allel.
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))
        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
    return genotypes, samples


def sort_samples(samples):
    sample_data = pd.read_csv(args.sample_data, sep="\t")
    sample_data["sampleID2"] = sample_data["sampleID"]
    sample_data.set_index("sampleID", inplace=True)
    samples = samples.astype("str")
    sample_data = sample_data.reindex(
        np.array(samples)
    )  # sort loc table so samples are in same order as vcf samples
    if not all(
        [sample_data["sampleID2"][x] == samples[x] for x in range(len(samples))]
    ):  # check that all sample names are present
        print("sample ordering failed! Check that sample IDs match the VCF.")
        sys.exit()
    locs = np.array(sample_data[["x", "y"]])
    print("loaded " + str(np.shape(genotypes)) + " genotypes\n\n")
    return (sample_data, locs)


# replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    print("imputing missing data")
    dc = genotypes.count_alleles()[:, 1]
    ac = genotypes.to_allele_counts()[:, :, 1]
    missingness = genotypes.is_missing()
    ninds = np.array([np.sum(x) for x in ~missingness])
    af = np.array([dc[x] / (2 * ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if missingness[i, j]:
                ac[i, j] = np.random.binomial(2, af[i])
    return ac


def filter_snps(genotypes):
    print("filtering SNPs")
    tmp = genotypes.count_alleles()
    biallel = tmp.is_biallelic()
    genotypes = genotypes[biallel, :, :]
    if not args.min_mac == 1:
        derived_counts = genotypes.count_alleles()[:, 1]
        ac_filter = [x >= args.min_mac for x in derived_counts]
        genotypes = genotypes[ac_filter, :, :]
    if args.impute_missing:
        ac = replace_md(genotypes)
    else:
        ac = genotypes.to_allele_counts()[:, :, 1]
    if not args.max_SNPs == None:
        ac = ac[np.random.choice(range(ac.shape[0]), args.max_SNPs, replace=False), :]
    print("running on " + str(len(ac)) + " genotypes after filtering\n\n\n")
    return ac


def normalize_locs(locs):
    meanlong = np.nanmean(locs[:, 0])
    sdlong = np.nanstd(locs[:, 0])
    meanlat = np.nanmean(locs[:, 1])
    sdlat = np.nanstd(locs[:, 1])
    locs = np.array(
        [[(x[0] - meanlong) / sdlong, (x[1] - meanlat) / sdlat] for x in locs]
    )
    return meanlong, sdlong, meanlat, sdlat, locs


def split_train_test(ac, locs, train_split=0.7, val_split=0.15):
    """
    Split the dataset into training, testing, and validation sets.

    Args:
        ac (np.ndarray): The array containing allele counts.
        locs (np.ndarray): The array containing locations.
        train_split (float): The proportion of the dataset to include in the train split.
        val_split (float): The proportion of the dataset to include in the validation split.

    Returns:
        tuple: A tuple containing indices and data for train, test, val, and pred sets.
    """
    # Determine train and prediction indices
    train_indices = np.argwhere(~np.isnan(locs[:, 0])).flatten()
    pred_indices = np.array([x for x in range(len(locs)) if x not in train_indices])

    # Shuffle the train indices
    np.random.shuffle(train_indices)

    # Determine the number of samples in each set
    num_train = int(len(train_indices) * train_split)
    num_val = int(len(train_indices) * val_split)

    # Split the indices
    train = train_indices[:num_train]
    val = train_indices[num_train : num_train + num_val]
    test = train_indices[num_train + num_val :]

    # Split the data
    traingen = np.transpose(ac[:, train])
    trainlocs = locs[train]
    testgen = np.transpose(ac[:, test])
    testlocs = locs[test]
    valgen = np.transpose(ac[:, val])
    vallocs = locs[val]
    predgen = np.transpose(ac[:, pred_indices])

    return (
        train,
        test,
        val,
        traingen,
        testgen,
        valgen,
        trainlocs,
        testlocs,
        vallocs,
        pred_indices,
        predgen,
    )


def load_network(traingen, dropout_prop):
    from tensorflow.keras import backend as K

    def euclidean_distance_loss(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.BatchNormalization(input_shape=(traingen.shape[1],)))
    for i in range(int(np.floor(args.nlayers / 2))):
        model.add(tf.keras.layers.Dense(args.width, activation="elu"))
    model.add(tf.keras.layers.Dropout(args.dropout_prop))
    for i in range(int(np.ceil(args.nlayers / 2))):
        model.add(tf.keras.layers.Dense(args.width, activation="elu"))
    model.add(tf.keras.layers.Dense(2))
    model.add(tf.keras.layers.Dense(2))
    model.compile(optimizer="Adam", loss=euclidean_distance_loss)
    return model


def load_callbacks(boot):
    if args.bootstrap or args.jacknife:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath="locator/" + args.out + "_boot" + str(boot) + "_weights.hdf5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            period=1,
        )
    else:
        checkpointer = tf.keras.callbacks.ModelCheckpoint(
            filepath="locator/" + args.out + "_weights.hdf5",
            verbose=args.keras_verbose,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            period=1,
        )
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=args.patience
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=int(args.patience / 6),
        verbose=args.keras_verbose,
        mode="auto",
        min_delta=0,
        cooldown=0,
        min_lr=0,
    )
    return checkpointer, earlystop, reducelr


@timer
def train_network(model, traingen, testgen, trainlocs, testlocs):
    history = model.fit(
        traingen,
        trainlocs,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=args.keras_verbose,
        validation_data=(testgen, testlocs),
        callbacks=[checkpointer, earlystop, reducelr],
    )
    if args.bootstrap or args.jacknife:
        model.load_weights(
            "locator/" + args.out + "_boot" + str(boot) + "_weights.hdf5"
        )
    else:
        model.load_weights("locator/" + args.out + "_weights.hdf5")
    return history, model


def predict_locs(
    model,
    predgen,
    sdlong,
    meanlong,
    sdlat,
    meanlat,
    testlocs,
    vallocs,
    pred,
    samples,
    testgen,
    valgen,
    verbose=True,
):
    if verbose == True:
        print("predicting locations...")
    prediction = model.predict(predgen)
    prediction = np.array(
        [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in prediction]
    )
    predout = pd.DataFrame(prediction)
    predout.columns = ["x", "y"]
    predout["sampleID"] = samples[pred]
    if args.bootstrap or args.jacknife:
        predout.to_csv(
            "locator/" + args.out + "_boot" + str(boot) + "_predlocs.txt", index=False
        )
        testlocs2 = np.array(
            [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in testlocs]
        )
    elif args.windows:
        predout.to_csv(
            "locator/"
            + args.out
            + "_"
            + str(i)
            + "-"
            + str(i + size - 1)
            + "_predlocs.txt",
            index=False,
        )  # this is dumb
        testlocs2 = np.array(
            [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in testlocs]
        )

        vallocs2 = np.array(
            [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in vallocs]
        )
    else:
        predout.to_csv("locator/" + args.out + "_predlocs.txt", index=False)
        testlocs2 = np.array(
            [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in testlocs]
        )
        vallocs2 = np.array(
            [[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in vallocs]
        )
    p2 = model.predict(testgen)  # print validation loss to screen
    p3 = model.predict(valgen)
    p2 = np.array([[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in p2])
    p3 = np.array([[x[0] * sdlong + meanlong, x[1] * sdlat + meanlat] for x in p3])
    r2_long = np.corrcoef(p2[:, 0], testlocs2[:, 0])[0][1] ** 2
    r2_lat = np.corrcoef(p2[:, 1], testlocs2[:, 1])[0][1] ** 2
    r2_long3 = np.corrcoef(p3[:, 0], vallocs2[:, 0])[0][1] ** 2
    r2_lat3 = np.corrcoef(p3[:, 1], vallocs2[:, 1])[0][1] ** 2
    mean_dist = np.mean(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    mean_dist3 = np.mean(
        [spatial.distance.euclidean(p3[x, :], vallocs2[x, :]) for x in range(len(p3))]
    )
    median_dist = np.median(
        [spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))]
    )
    median_dist3 = np.median(
        [spatial.distance.euclidean(p3[x, :], vallocs2[x, :]) for x in range(len(p3))]
    )
    dists = [
        spatial.distance.euclidean(p2[x, :], testlocs2[x, :]) for x in range(len(p2))
    ]
    dists3 = [
        spatial.distance.euclidean(p3[x, :], vallocs2[x, :]) for x in range(len(p3))
    ]

    def mad(data):
        return np.median(np.abs(data - np.median(data)))

    def coefficient_of_variation(data):
        return np.std(data) / np.mean(data)

    def within_threshold(data, threshold):
        return np.mean(data < threshold)

    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    # Calculate Haversine error for each pair of points
    haversine_errors = haversine_distances_agg(testlocs2, p2, np.array)
    haversine_errors3 = haversine_distances_agg(vallocs2, p3, np.array)

    stats = get_all_stats(
        p2, testlocs2, mad, coefficient_of_variation, within_threshold
    )
    stats3 = get_all_stats(
        p3, vallocs2, mad, coefficient_of_variation, within_threshold
    )

    print_stats_to_logger(**stats)
    print_stats_to_logger(**stats3)

    plotting.plot_geographic_error_distribution(
        testlocs2,
        p2,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        dataset="validation",
    )
    plotting.plot_geographic_error_distribution(
        vallocs2,
        p3,
        url="https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip",
        dataset="test",
    )

    plotting.polynomial_regression_plot(testlocs2, p2, dataset="validation")
    plotting.polynomial_regression_plot(vallocs2, p3, dataset="test")

    plotting.plot_cumulative_error_distribution(
        haversine_errors,
        "original_locator_2_ecdf_validation_plot.pdf",
        stats["percentiles"],
        stats["median_dist"],
        stats["mean_dist"],
    )
    plotting.plot_cumulative_error_distribution(
        haversine_errors3,
        "original_locator_2_ecdf_test_plot.pdf",
        stats3["percentiles"],
        stats3["median_dist"],
        stats3["mean_dist"],
    )

    plotting.plot_error_distribution(
        haversine_errors, "error_distribution_validation.pdf"
    )
    plotting.plot_error_distribution(haversine_errors3, "error_distribution_test.pdf")

    plotting.plot_zscores(
        stats["z_scores"],
        haversine_errors,
        "original_locator_2_zscores_validation.pdf",
    )
    plotting.plot_zscores(
        stats3["z_scores"],
        haversine_errors3,
        "original_locator_2_zscores_test.pdf",
    )

    stats["mean_absolute_z_score"] = np.mean(np.abs(stats["z_scores"]))
    stats3["mean_absolute_z_score"] = np.mean(np.abs(stats3["z_scores"]))
    stats.pop("z_scores")
    stats3.pop("z_scores")

    jstats = {}
    for k, v in stats.items():
        if not isinstance(v, np.ndarray):
            jstats[k] = v
    with open("locator/test/original_locator_2_validation_metrics.json", "w") as f:
        json.dump(jstats, f, indent=2)
    jstats3 = {}
    for k, v in stats3.items():
        if not isinstance(v, np.ndarray):
            jstats3[k] = v
    with open("locator/test/original_locator_2_test_metrics.json", "w") as f:
        json.dump(jstats3, f, indent=2)

    if verbose == True:
        print(
            "R2(x)="
            + str(r2_long)
            + "\nR2(y)="
            + str(r2_lat)
            + "\n"
            + "mean validation error "
            + str(mean_dist)
            + "\n"
            + "median validation error "
            + str(median_dist)
            + "\n"
        )
        print(
            "R2(x)="
            + str(r2_long3)
            + "\nR2(y)="
            + str(r2_lat3)
            + "\n"
            + "mean validation error "
            + str(mean_dist3)
            + "\n"
            + "median validation error "
            + str(median_dist3)
            + "\n"
        )
    hist = pd.DataFrame(history.history)
    hist.to_csv("locator/" + args.out + "_history.txt", sep="\t", index=False)
    return dists3


def get_all_stats(
    predictions, ground_truth, mad, coefficient_of_variation, within_threshold
):
    rmse = calculate_rmse(predictions, ground_truth)
    mean_dist = haversine_distances_agg(ground_truth, predictions, np.mean)
    median_dist = haversine_distances_agg(ground_truth, predictions, np.median)
    std_dist = haversine_distances_agg(ground_truth, predictions, np.std)

    haversine_errors = haversine_distances_agg(ground_truth, predictions, np.array)

    # Ideal distances array - all zeros indicating no error
    ideal_distances = np.zeros_like(haversine_errors)

    (
        spearman_corr_haversine,
        spearman_corr_x,
        spearman_corr_y,
        spearman_p_value_haversine,
        spearman_p_value_x,
        spearman_p_value_y,
    ) = get_correlation_coef(
        predictions, ground_truth, haversine_errors, ideal_distances, spearmanr
    )
    (
        pearson_corr_haversine,
        pearson_corr_x,
        pearson_corr_y,
        pearson_p_value_haversine,
        pearson_p_value_x,
        pearson_p_value_y,
    ) = get_correlation_coef(
        predictions, ground_truth, haversine_errors, ideal_distances, pearsonr
    )

    rho, rho_p = spearmanr(predictions.ravel(), ground_truth.ravel())

    # Calculate median absolute deviation for Haversine distances
    haversine_mad = mad(haversine_errors)

    cv = coefficient_of_variation(haversine_errors)

    # Inter-quartile range.
    iqr = np.percentile(haversine_errors, 75) - np.percentile(haversine_errors, 25)

    percentiles = np.percentile(haversine_errors, [25, 50, 75])

    # Percentage of predictions within <N> km error
    percentage_within_20km = within_threshold(haversine_errors, 25) * 100
    percentage_within_50km = within_threshold(haversine_errors, 50) * 100
    percentage_within_75km = within_threshold(haversine_errors, 75) * 100

    z_scores = (haversine_errors - np.mean(haversine_errors)) / np.std(haversine_errors)

    mean_absolute_z_score = np.mean(np.abs(z_scores))

    haversine_rmse = mean_squared_error(
        haversine_errors.reshape(-1, 1),
        np.zeros_like(haversine_errors).reshape(-1, 1),
        squared=False,
    )

    # 0 is best, negative means overestimations, positive means
    # underestimations
    ks, pval, skew = kstest(ground_truth, predictions)
    return create_dictionary(
        rmse,
        mean_dist,
        median_dist,
        std_dist,
        spearman_corr_haversine,
        spearman_corr_x,
        spearman_corr_y,
        spearman_p_value_haversine,
        spearman_p_value_x,
        spearman_p_value_y,
        pearson_corr_haversine,
        pearson_corr_x,
        pearson_corr_y,
        pearson_p_value_haversine,
        pearson_p_value_x,
        pearson_p_value_y,
        rho,
        rho_p,
        haversine_mad,
        cv,
        iqr,
        percentiles,
        percentage_within_20km,
        percentage_within_50km,
        percentage_within_75km,
        z_scores,
        mean_absolute_z_score,
        haversine_rmse,
        ks,
        pval,
        skew,
    )


def create_dictionary(
    rmse,
    mean_dist,
    median_dist,
    std_dist,
    spearman_corr_haversine,
    spearman_corr_x,
    spearman_corr_y,
    spearman_p_value_haversine,
    spearman_p_value_x,
    spearman_p_value_y,
    pearson_corr_haversine,
    pearson_corr_x,
    pearson_corr_y,
    pearson_p_value_haversine,
    pearson_p_value_x,
    pearson_p_value_y,
    rho,
    rho_p,
    haversine_mad,
    cv,
    iqr,
    percentiles,
    percentage_within_20km,
    percentage_within_50km,
    percentage_within_75km,
    z_scores,
    mean_absolute_z_score,
    haversine_rmse,
    ks,
    pval,
    skew,
):
    """
    Create a hard-coded dictionary with specified metrics.

    Args:
        rmse, mean_dist, median_dist, std_dist: Statistical metrics.
        spearman_corr_haversine, spearman_corr_x, spearman_corr_y: Spearman correlations.
        spearman_p_value_haversine, spearman_p_value_x, spearman_p_value_y: Spearman p-values.
        pearson_corr_haversine, pearson_corr_x, pearson_corr_y: Pearson correlations.
        pearson_p_value_haversine, pearson_p_value_x, pearson_p_value_y: Pearson p-values.
        rho, rho_p: Rho statistics.
        haversine_mad, cv, iqr: Distance measures.
        percentiles: Percentile values.
        percentage_within_20km, percentage_within_50km, percentage_within_75km: Distance thresholds.
        z_scores, mean_absolute_z_score: Z-score metrics.
        haversine_rmse, ks, pval, skew: Various statistical metrics.

    Returns:
    dict: Dictionary with metric names as keys and their values.
    """
    return {
        "rmse": rmse,
        "mean_dist": mean_dist,
        "median_dist": median_dist,
        "std_dist": std_dist,
        "spearman_corr_haversine": spearman_corr_haversine,
        "spearman_corr_x": spearman_corr_x,
        "spearman_corr_y": spearman_corr_y,
        "spearman_p_value_haversine": spearman_p_value_haversine,
        "spearman_p_value_x": spearman_p_value_x,
        "spearman_p_value_y": spearman_p_value_y,
        "pearson_corr_haversine": pearson_corr_haversine,
        "pearson_corr_x": pearson_corr_x,
        "pearson_corr_y": pearson_corr_y,
        "pearson_p_value_haversine": pearson_p_value_haversine,
        "pearson_p_value_x": pearson_p_value_x,
        "pearson_p_value_y": pearson_p_value_y,
        "rho": rho,
        "rho_p": rho_p,
        "haversine_mad": haversine_mad,
        "cv": cv,
        "iqr": iqr,
        "percentiles": percentiles,
        "percentage_within_20km": percentage_within_20km,
        "percentage_within_50km": percentage_within_50km,
        "percentage_within_75km": percentage_within_75km,
        "z_scores": z_scores,
        "mean_absolute_z_score": mean_absolute_z_score,
        "haversine_rmse": haversine_rmse,
        "ks": ks,
        "pval": pval,
        "skew": skew,
    }


def print_stats_to_logger(
    rmse,
    mean_dist,
    median_dist,
    std_dist,
    spearman_corr_haversine,
    spearman_corr_x,
    spearman_corr_y,
    spearman_p_value_haversine,
    spearman_p_value_x,
    spearman_p_value_y,
    pearson_corr_haversine,
    pearson_corr_x,
    pearson_corr_y,
    pearson_p_value_haversine,
    pearson_p_value_x,
    pearson_p_value_y,
    rho,
    rho_p,
    haversine_mad,
    cv,
    iqr,
    percentiles,
    percentage_within_20km,
    percentage_within_50km,
    percentage_within_75km,
    z_scores,
    mean_absolute_z_score,
    haversine_rmse,
    ks,
    pval,
    skew,
):
    logger.info(f"Validation Haversine Error (km) = {mean_dist}")
    logger.info(f"Median Validation Error (km) = {median_dist}")
    logger.info(f"Standard deviation for Haversine Error (km) = {std_dist}")

    logger.info(f"Root Mean Squared Error (km) = {haversine_rmse}")
    logger.info(f"Median Absolute Deviation of Prediction Error (km) = {haversine_mad}")
    logger.info(f"Coeffiecient of Variation for Prediction Error = {cv}")
    logger.info(f"Interquartile Range of Prediction Error (km) = {iqr}")

    for perc, output in zip([25, 50, 75], percentiles):
        logger.info(f"{perc} percentile of prediction error (km) = {output}")

    logger.info(
        f"Percentage of samples with error within 20 km = {percentage_within_20km}"
    )
    logger.info(
        f"Percentage of samples with error within 50 km = {percentage_within_50km}"
    )
    logger.info(
        f"Percentage of samples with error within 75 km = {percentage_within_75km}"
    )

    logger.info(
        f"Mean Absolute Z-scores of Prediction Error (km) = {mean_absolute_z_score}"
    )

    logger.info(f"Spearman's Correlation Coefficient = {rho}, P-value = {rho_p}")
    logger.info(
        f"Spearman's Correlation Coefficient for Longitude = {spearman_corr_x}, P-value = {spearman_p_value_x}"
    )
    logger.info(
        f"Spearman's Correlation Coefficient for Latitude = {spearman_corr_y}, P-value = {spearman_p_value_y}"
    )
    logger.info(
        f"Pearson's Correlation Coefficient for Longitude = {pearson_corr_x}, P-value = {pearson_p_value_x}"
    )
    logger.info(
        f"Pearson's Correlation Coefficient for Latitude = {pearson_corr_y}, P-value = {pearson_p_value_y}"
    )

    # 0 is best, positive means more undeerestimations
    # negative means more overestimations.
    logger.info(f"Skewness = {skew}")

    # Goodness of fit test.
    # Small P-value means poor fit.
    # I.e., significantly deviates from reference distribution.
    logger.info(f"Kolmogorov-Smirnov Test = {ks}, P-value = {pval}")


def get_correlation_coef(
    predictions, ground_truth, haversine_errors, ideal_distances, corr_func
):
    corr_haversine, p_value_haversine = spearmanr(haversine_errors, ideal_distances)

    corr_x, p_value_x = corr_func(predictions[:, 0], ground_truth[:, 0])

    corr_y, p_value_y = corr_func(predictions[:, 1], ground_truth[:, 1])
    return corr_haversine, corr_x, corr_y, p_value_haversine, p_value_x, p_value_y


### windows ###
if args.windows:
    callset = zarr.open_group(args.zarr, mode="r")
    gt = callset["calldata/GT"]
    samples = callset["samples"][:]
    positions = np.array(callset["variants/POS"])
    start = int(args.window_start)
    if args.window_stop == None:
        stop = np.max(positions)
    else:
        stop = int(args.window_stop)
    size = int(args.window_size)
    for i in np.arange(start, stop, size):
        mask = np.logical_and(positions >= i, positions < i + size)
        a = np.min(np.argwhere(mask))
        b = np.max(np.argwhere(mask))
        print(a, b)
        genotypes = allel.GenotypeArray(gt[a:b, :, :])
        sample_data, locs = sort_samples(samples)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            val,
            traingen,
            testgen,
            valgen,
            trainlocs,
            testlocs,
            vallocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network(traingen, args.dropout_prop)
        t1 = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            vallocs,
            pred,
            samples,
            testgen,
            valgen,
        )
        plotting.plot_history(history.history["loss"], history.history["val_loss"])
        if not args.keep_weights:
            subprocess.run("rm " + "locator/" + args.out + "_weights.hdf5", shell=True)
        t2 = time.time()
        elapsed = t2 - t1
        print("run time " + str(elapsed / 60) + " minutes")
else:
    if not args.bootstrap and not args.jacknife:
        boot = None
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            val,
            traingen,
            testgen,
            valgen,
            trainlocs,
            testlocs,
            vallocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network(traingen, args.dropout_prop)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            vallocs,
            pred,
            samples,
            testgen,
            valgen,
        )
        plotting.plot_history(history.history["loss"], history.history["val_loss"])
        if not args.keep_weights:
            subprocess.run("rm " + "locator/" + args.out + "_weights.hdf5", shell=True)
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
    elif args.bootstrap:
        boot = "FULL"
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks("FULL")
        (
            train,
            test,
            val,
            traingen,
            testgen,
            valgen,
            trainlocs,
            testlocs,
            vallocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network(traingen, args.dropout_prop)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            vallocs,
            pred,
            samples,
            testgen,
            valgen,
        )
        plotting.plot_history(history.history["loss"], history.history["val_loss"])
        if not args.keep_weights:
            subprocess.run(
                "rm " + "locator/" + args.out + "_bootFULL_weights.hdf5", shell=True
            )
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        for boot in range(args.nboots):
            np.random.seed(np.random.choice(range(int(1e6)), 1))
            checkpointer, earlystop, reducelr = load_callbacks(boot)
            print("starting bootstrap " + str(boot))
            traingen2 = copy.deepcopy(traingen)
            testgen2 = copy.deepcopy(testgen)
            predgen2 = copy.deepcopy(predgen)
            valgen2 = copy.deepcopy(valgen)
            site_order = np.random.choice(
                traingen2.shape[1], traingen2.shape[1], replace=True
            )
            traingen2 = traingen2[:, site_order]
            testgen2 = testgen2[:, site_order]
            predgen2 = predgen2[:, site_order]
            valgen2 = valgen2[:, site_order]
            model = load_network(traingen2, args.dropout_prop)
            start = time.time()
            history, model = train_network(
                model, traingen2, testgen2, trainlocs, testlocs
            )
            dists = predict_locs(
                model,
                predgen2,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                vallocs,
                pred,
                samples,
                testgen2,
                valgen2,
            )
            plotting.plot_history(history.history["loss"], history.history["val_loss"])
            if not args.keep_weights:
                subprocess.run(
                    "rm "
                    + "locator/"
                    + args.out
                    + "_boot"
                    + str(boot)
                    + "_weights.hdf5",
                    shell=True,
                )
            end = time.time()
            elapsed = end - start
            K.clear_session()
            print("run time " + str(elapsed / 60) + " minutes\n\n")
    elif args.jacknife:
        boot = "FULL"
        genotypes, samples = load_genotypes()
        sample_data, locs = sort_samples(samples)
        meanlong, sdlong, meanlat, sdlat, locs = normalize_locs(locs)
        ac = filter_snps(genotypes)
        checkpointer, earlystop, reducelr = load_callbacks(boot)
        (
            train,
            test,
            val,
            traingen,
            testgen,
            valgen,
            trainlocs,
            testlocs,
            vallocs,
            pred,
            predgen,
        ) = split_train_test(ac, locs)
        model = load_network(traingen, args.dropout_prop)
        start = time.time()
        history, model = train_network(model, traingen, testgen, trainlocs, testlocs)
        dists = predict_locs(
            model,
            predgen,
            sdlong,
            meanlong,
            sdlat,
            meanlat,
            testlocs,
            vallocs,
            pred,
            samples,
            testgen,
            valgen,
        )

        plotting.plot_history(history.history["loss"], history.history["val_loss"])
        end = time.time()
        elapsed = end - start
        print("run time " + str(elapsed / 60) + " minutes")
        print("starting jacknife resampling")
        af = []
        for i in tqdm(range(ac.shape[0])):
            af.append(sum(ac[i, :]) / (ac.shape[1] * 2))
        af = np.array(af)
        for boot in tqdm(range(args.nboots)):
            checkpointer, earlystop, reducelr = load_callbacks(boot)
            pg = copy.deepcopy(predgen)  # this asshole
            sites_to_remove = np.random.choice(
                pg.shape[1], int(pg.shape[1] * args.jacknife_prop), replace=False
            )  # treat X% of sites as missing data
            for i in sites_to_remove:
                pg[:, i] = np.random.binomial(2, af[i], pg.shape[0])
                # pg[:,i]=af[i]
            dists = predict_locs(
                model,
                pg,
                sdlong,
                meanlong,
                sdlat,
                meanlat,
                testlocs,
                vallocs,
                pred,
                samples,
                testgen,
                valgen,
                verbose=False,
            )  # TODO: check testgen behavior for printing R2 to screen with jacknife in predict mode
        if not args.keep_weights:
            subprocess.run(
                "rm " + "locator/" + args.out + "_bootFULL_weights.hdf5", shell=True
            )

    Path("locator/benchmarking").mkdir(parents=True, exist_ok=True)
    save_execution_times("locator/benchmarking/execution_times.csv")
    execution_times.clear()
