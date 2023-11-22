import random
from collections import Counter

import msprime
import pysam
import numpy as np
import pandas as pd
import demes
import pprint
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA


from snpio import GenotypeData
from snpio.simulators.simulate import SNPulator, SNPulatorConfig, SNPulatoRate


import numpy as np
import msprime


def create_migration_matrix(n_pops, min_rate, max_rate):
    """
    Create a migration matrix for populations based on their distances.

    Args:
        n_pops (int): Number of populations.
        min_rate (float): Minimum migration rate.
        max_rate (float): Maximum migration rate.

    Returns:
        np.array: A migration matrix of size n_pops x n_pops.
    """
    # Create a matrix of distances between populations
    pop_indices = np.arange(1, n_pops + 1)
    distance_matrix = np.abs(pop_indices[:, None] - pop_indices).astype(float)

    # Calculate migration rates based on distance
    # Avoid division by zero for same population by replacing 0 with np.inf temporarily
    distance_matrix[distance_matrix == 0] = np.inf
    migration_matrix = max_rate / distance_matrix
    migration_matrix[
        np.isinf(migration_matrix)
    ] = 0  # Set migration rate to 0 for same population

    # Ensure migration rates do not fall below the minimum rate
    migration_matrix[migration_matrix < min_rate] = min_rate

    return migration_matrix


def demographic_model(
    n_pops=8,
    num_samples=100,
    min_log_mig=-1.5,
    max_log_mig=1.5,
    growth_rate=0.01,
    growth_rate_anc=-0.01,
    time=45,
    recovery_ne=25000,
    bottleneck_ne=50,
    debug=False,
):
    def log_to_linear(ln):
        return np.exp(ln) / (2 * bottleneck_ne)

    # Initialize demography
    demography = msprime.Demography()

    # Add populations
    for pop in range(1, n_pops + 1):
        demography.add_population(name=f"Pop{pop}", initial_size=recovery_ne)

    # Assign samples to these populations
    samples = [
        msprime.SampleSet(population=pop, num_samples=num_samples)
        for pop in range(n_pops)
    ]

    # Create and set migration matrix
    migration_matrix = create_migration_matrix(
        n_pops, log_to_linear(min_log_mig), log_to_linear(max_log_mig)
    )
    for i in range(n_pops):
        for j in range(n_pops):
            if i != j:
                demography.set_symmetric_migration_rate(
                    [f"Pop{i+1}", f"Pop{j+1}"], migration_matrix[i, j]
                )

    # Set population parameters changes
    for pop in range(1, n_pops + 1):
        demography.add_population_parameters_change(
            time=0,
            initial_size=recovery_ne,
            growth_rate=growth_rate,
            population=f"Pop{pop}",
        )
        demography.add_population_parameters_change(
            time=time,
            initial_size=bottleneck_ne,
            growth_rate=growth_rate_anc,
            population=f"Pop{pop}",
        )

    demography.sort_events()

    if debug:
        print(demography.debug())

    return demography


def introduce_missing_data(input_vcf, output_vcf, missing_ratio=0.2):
    # Open the VCF file
    vcf_in = pysam.VariantFile(input_vcf, "r")
    vcf_out = pysam.VariantFile(output_vcf, "w", header=vcf_in.header)

    # Calculate total number of genotype entries and determine how many to replace
    total_entries = sum(len(record.samples) for record in vcf_in)
    num_missing = int(total_entries * missing_ratio)

    # Generate random positions to replace with missing data
    missing_positions = set(random.sample(range(total_entries), num_missing))

    # Reset file pointer to the beginning of the file
    vcf_in.seek(0)

    # Counter for current position
    current_pos = 0

    for record in vcf_in:
        for sample in record.samples.values():
            # Replace with missing data if this position is selected
            if current_pos in missing_positions:
                sample["GT"] = (None, None)
            current_pos += 1

        # Write the modified record to the new VCF
        vcf_out.write(record)

    vcf_in.close()
    vcf_out.close()


# Function to generate random chromosome IDs
def generate_chromosome_ids(num_snps):
    return [f"CM{str(i+1).zfill(4)}.{i+1}" for i in range(num_snps)]


# Function to generate random positions
def generate_random_positions(num_snps):
    return np.random.randint(1, 1_000_000, size=num_snps)


def get_most_common_alleles(df, snp_id, hets):
    """
    Get the most common alleles from a DataFrame column, excluding specified heterozygous alleles.

    Args:
        df (pd.DataFrame): DataFrame containing SNP data.
        snp_id (str): Column name in df representing the SNP.
        hets (set): Set of alleles to be excluded from the count.

    Returns:
        tuple: A tuple containing the most common allele and the second most common allele. If no alleles meet the criteria, return ('.', '.').
    """
    # Filter out alleles in 'hets'
    filtered_alleles = df[snp_id].dropna().apply(lambda x: x if x not in hets else None)

    # Count allele frequencies excluding those in 'hets'
    allele_counts = Counter(filtered_alleles.dropna())
    common_alleles = [
        allele for allele, count in allele_counts.most_common() if allele not in hets
    ]

    # Determine the reference and alternate alleles
    ref = common_alleles[0] if common_alleles else "."
    alt = common_alleles[1] if len(common_alleles) > 1 else "."

    return ref, alt


def df2vcf(filename):
    df = pd.read_csv("data/simulated_unlinked_big.csv")

    # Generate chromosome IDs and positions
    chromosome_ids = generate_chromosome_ids(len(df.columns))
    positions = generate_random_positions(len(df.columns))

    # Sample names (assuming rows are samples)
    sample_names = "sample" + df.index.astype(str)
    sample_names = sample_names.tolist()

    # Write to VCF file
    with open("data/simulated_unlinked_big.vcf", "w") as vcf:
        # Write VCF header
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write("##FORMAT=<ID=GT,Number=1,Type=String,Description='Genotype'>\n")
        vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        vcf.write("\t".join(sample_names) + "\n")

        allele_freqs = []
        allele_freqs2 = []
        df2 = pd.DataFrame(np.zeros_like(df.to_numpy(), dtype=str), columns=df.columns)
        # Write genotype data for each SNP
        for i, snp_id in enumerate(df.columns):
            chrom = chromosome_ids[i]
            pos = positions[i]
            hets = {"R", "Y", "M", "K", "S", "W", "H", "B", "V", "D"}
            ref, alt = get_most_common_alleles(df, snp_id, hets)
            freqs = get_allele_frequencies(df, snp_id, hets)
            allele_freqs.append(freqs)

            # Convert genotypes to '0/0', '0/1', '1/1' format
            def convert_to_vcf_genotype(allele):
                if allele == ref:
                    return "0/0"
                elif allele == alt:
                    return "1/1"
                elif allele in hets:
                    return "0/1"
                else:
                    return "./."

            genotype_data = "\t".join(
                df[snp_id].apply(convert_to_vcf_genotype).astype(str)
            )
            vcf_line = f"{chrom}\t{pos}\t{snp_id}\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{genotype_data}\n"
            vcf.write(vcf_line)

            df2[snp_id] = df[snp_id].apply(convert_to_vcf_genotype).astype(str)
            freqs2 = get_allele_frequencies(df2, snp_id, hets)
            allele_freqs2.append(freqs2)

    return allele_freqs, allele_freqs2


def sim_seqs(
    aln_filename,
    popmap_filename,
    demes_graph,
    mutation_rate=1e-8,
    seq_length=100,
    n_pops=4,
    n_samples=200,
    record_migrations=True,
    load_pkl=False,
):
    genotype_data = GenotypeData(
        filename=aln_filename,
        popmapfile=popmap_filename,
        force_popmap=True,
    )

    config = {
        "mutation_rate": mutation_rate,
        "demes_graph": demes_graph,
        "sequence_length": seq_length,
        "record_migrations": record_migrations,
        "mutation_model": msprime.GTR,
        "recombination_rate": 0.01,  # unlinked.
    }

    if load_pkl:
        with open("data/sim_snp_data_big.pkl", "rb") as fin:
            genotype_data_sim = pickle.load(fin)
    else:
        snprate = SNPulatoRate(genotype_data, 45)

        snp_config = config

        snpconfig = SNPulatorConfig(**snp_config)
        rate = snprate.calculate_rate(model="GTR")
        snpconfig.update_mutation_rate(rate)

        print(f"Updated mutation rate: {rate}")

        # snpconfig.update_mutation_rate(rate)
        snp = SNPulator(genotype_data, snpconfig)
        genotype_data_sim = snp.simulate_alignment(
            sample_sizes=[n_samples // n_pops for x in range(1, n_pops + 1)],
            populations=[f"Pop{i}" for i in range(1, n_pops + 1)],
        )

        with open("data/sim_snp_data_big.pkl", "wb") as fout:
            pickle.dump(genotype_data_sim.snp_data, fout)

    return genotype_data_sim


def get_allele_frequencies(df, snp_id, hets):
    """
    Get allele frequencies from a DataFrame column, excluding specified heterozygous alleles.

    Args:
        df (pd.DataFrame): DataFrame containing SNP data.
        snp_id (str): Column name in df representing the SNP.
        hets (set): Set of alleles to be excluded from the count.

    Returns:
        dict: A dictionary containing allele frequencies, excluding alleles in 'hets'.
    """
    # Filter out alleles in 'hets'
    filtered_alleles = df[snp_id].dropna().apply(lambda x: x if x not in hets else None)

    # Count allele frequencies excluding those in 'hets'
    allele_counts = Counter(filtered_alleles.dropna())

    # Convert counts to frequencies
    total_count = sum(allele_counts.values())
    allele_frequencies = {
        allele: count / total_count for allele, count in allele_counts.items()
    }

    return allele_frequencies


def plot_combined_allele_frequencies(allele_freq_list, filename, title, show=False):
    """
    Plot the combined distribution of allele frequencies from multiple dictionaries.

    Args:
        allele_freq_list (list): A list of dictionaries containing allele frequencies.
    """
    # Combine all allele frequencies into a single DataFrame
    combined_df = pd.DataFrame(allele_freq_list).fillna(0)

    # Calculate mean frequency for each allele across all dictionaries
    mean_frequencies = combined_df.mean(axis=0).sort_values(ascending=False)

    # Create a bar plot
    plt.figure(figsize=(12, 8))
    mean_frequencies.plot(kind="bar", color="skyblue")
    plt.xlabel("Allele")
    plt.ylabel("Average Frequency")
    plt.title("Combined Allele Frequencies Distribution")
    plt.xticks(rotation=45)

    if show:
        plt.show()
    plt.savefig(filename, facecolor="white", bbox_inches="tight")


def plot_allele_kde(allele_freq_list, filename, title, show=False):
    """
    Plot a KDE of allele frequencies with each allele as a separate hue group.

    Args:
        allele_freq_list (list): A list of dictionaries containing allele frequencies.
    """
    # Convert the list of dictionaries to a long-format DataFrame
    df_long = pd.DataFrame(allele_freq_list).melt(
        var_name="Allele", value_name="Frequency"
    )

    # Plotting KDE plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df_long, x="Frequency", hue="Allele", fill=True)
    plt.title("Density of Allele Frequencies")
    plt.xlabel("Frequency")
    plt.ylabel("Density")

    if show:
        plt.show()
    plt.savefig(filename, facecolor="white", bbox_inches="tight")


def main():
    num_samples = 1400
    num_loci = 2000
    n_pops = 8
    mutation_rate = 1.8e-4
    record_migrations = False
    ne_high = 25000
    ne_low = 50
    debug = True

    # Run the simulation function
    demography = demographic_model(
        n_pops=n_pops,
        num_samples=num_samples,
        min_log_mig=-1.5,
        max_log_mig=1.48,
        growth_rate=0.135,
        growth_rate_anc=0.0,
        recovery_ne=ne_high,
        bottleneck_ne=ne_low,
        debug=debug,
    )

    g = demography.to_demes()
    demes.dump(g, "data/demography.yaml", simplified=False)

    genotype_data_sim = sim_seqs(
        aln_filename="data/phase6_gtseq_subset.vcf.gz",
        popmap_filename="data/wtd_popmap.txt",
        demes_graph="data/demography.yaml",
        mutation_rate=mutation_rate,
        seq_length=num_loci,
        n_pops=n_pops,
        n_samples=num_samples,
        record_migrations=record_migrations,
        load_pkl=True,
    )

    try:
        df = pd.DataFrame(genotype_data_sim.snp_data)
    except AttributeError:
        df = pd.DataFrame(genotype_data_sim)
    df.to_csv("data/simulated_unlinked_big.csv", header=False, index=False)

    af, af2 = df2vcf("data/simulated_unlinked_big.csv")
    plot_combined_allele_frequencies(
        af2, "data/af_sim_bar.png", title="(Simulated Data)", show=False
    )
    plot_allele_kde(af2, "data/af_sim_kde.png", title="(Simulated Data)", show=False)

    data_orig = GenotypeData(
        filename="data/phase6_gtseq_subset.vcf.gz",
        popmapfile="data/wtd_popmap.txt",
        force_popmap=True,
    )

    df_orig = pd.DataFrame(data_orig.snp_data)
    df_orig.to_csv("data/df_orig.csv", header=False, index=False)
    af_orig, af_orig2 = df2vcf("data/df_orig.csv")

    plot_combined_allele_frequencies(
        af_orig2, "data/af_orig_bar.png", title="(Original Data)", show=False
    )
    plot_allele_kde(
        af_orig2, "data/af_orig_kde.png", title="(Original Data)", show=False
    )


if __name__ == "__main__":
    main()
