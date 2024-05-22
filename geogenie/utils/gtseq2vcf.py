import argparse
import gzip
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pysam
import seaborn as sns


class GTseqToVCF:
    def __init__(self, filepath, str2drop=None):
        self.filepath = filepath
        self.data = None
        self.header = []
        self.vcf_data = []
        self.snp_ids = None
        self.sample_ids = None
        self.header_key = "Sample"

        self.metadata_cols = [
            "Raw Reads",
            "On-Target Reads",
            "%On-Target",
            "%GT",
            "IFI",
        ]

        self.str2drop = str2drop

    def load_data(self):
        # Load the data
        self.data = pd.read_csv(self.filepath, header=0)

        # Remove metadata columns and store in self.metadata
        self.metadata = pd.concat(
            [self.data.pop(x) for x in self.metadata_cols], axis=1
        )

        self.data.columns = self.data.columns.str.strip()

        if self.str2drop is not None:
            if isinstance(self.str2drop, list):
                for to_drop in self.str2drop:
                    self.data = self.data.loc[
                        :, ~self.data.columns.str.startswith(to_drop)
                    ]
                    self.data = self.data.loc[
                        :, ~self.data.columns.str.contains(to_drop)
                    ]
            elif isinstance(self.str2drop, str):
                self.data = self.data.loc[
                    :, ~self.data.columns.str.startswith(self.str2drop)
                ]
                self.data = self.data.loc[
                    :, ~self.data.columns.str.contains(self.str2drop)
                ]

            else:
                raise TypeError(
                    f"str2drop must be a list or str, but got: {type(self.str2drop)}"
                )

    def parse_sample_column(self):
        """
        Parses the column headers into separate 'CHROM' and 'POS' columns, and reorders the DataFrame to have 'CHROM' and 'POS' as the first two columns.
        Validates the 'POS' to be strictly numeric and the genotype columns to contain valid allele pairs. Assumes 'Sample' is the index of the DataFrame.

        Returns:
            None
        """

        # Extract 'CHROM' and 'POS' from the column headers\
        chrom_pos_split = [col.split("_", 1) for col in self.data.columns]
        chrom = [x[0] for x in chrom_pos_split if "Sample" not in x]
        pos = [x[1] for x in chrom_pos_split if "Sample" not in x]

        self.chrom = chrom
        self.pos = pos

        # Now, set the first column as index and transpose the DataFrame
        self.data = self.data.set_index(self.data.columns[0]).T
        self.data.insert(0, "CHROM", chrom)
        self.data.insert(1, "POS", pos)

        # After transposition, create a MultiIndex from the first two rows (which contain 'CHROM' and 'POS')
        self.data.index = pd.MultiIndex.from_arrays(
            [self.data.iloc[:, 0], self.data.iloc[:, 1]], names=("CHROM", "POS")
        )

        # Drop the now redundant first two rows
        self.data = self.data.drop(["CHROM", "POS"], axis=1)

        # Validate the 'POS' to ensure all values are numeric
        if not self.data.index.get_level_values("POS").str.isnumeric().all():
            raise ValueError("Non-numeric values found in 'POS' index level.")

        # Define valid allele pairs
        valid_alleles = {
            "AA",
            "TT",
            "CC",
            "GG",
            "AT",
            "TA",
            "AC",
            "CA",
            "AG",
            "GA",
            "TC",
            "CT",
            "TG",
            "GT",
            "CG",
            "GC",
            "--",
            "NN",
            "00",
        }

        # Validate the genotype columns to ensure they contain valid allele
        # pairs
        for sample in self.data:
            if (
                not self.data[sample]
                .apply(lambda x: x in valid_alleles or x == "00" or x == "0")
                .all()
            ):
                raise ValueError(
                    f"Invalid alleles found in sample {sample}: {self.data[sample].tolist()}"
                )

        self.data = self.data.T

    def calculate_ref_alt_alleles(self):
        """
        Calculates the reference (REF) and alternate (ALT) alleles for each SNP across all samples. Assumes the DataFrame has 'CHROM' and 'POS' as a MultiIndex for the columns and the rest of the columns are SNP genotypes.
        """

        # Define a helper function to calculate REF and ALT alleles for a series of genotypes
        def get_ref_alt(genotypes):
            # Flatten the genotype pairs and count occurrences, excluding missing data
            allele_counts = Counter(
                "".join(genotypes).replace("--", "").replace("NN", "").replace("00", "")
            )
            if allele_counts:
                # Get the most common alleles
                common_alleles = allele_counts.most_common()
                ref_allele = common_alleles[0][0] if common_alleles else None
                alt_allele = common_alleles[1][0] if len(common_alleles) > 1 else "."
            else:
                # No valid alleles present
                ref_allele, alt_allele = None, "."
            return ref_allele, alt_allele

        # Apply the helper function to each SNP column to calculate REF and ALT alleles
        ref_alt_alleles = self.data.apply(get_ref_alt)

        # Split the tuples of REF and ALT alleles into separate DataFrames
        refs, alts = zip(ref_alt_alleles.to_numpy())

        self.data = self.data.T

        # Assign the REF and ALT alleles to the corresponding MultiIndex levels
        self.data = self.data.assign(REF=refs[0], ALT=alts[0])

    def join_multiindex(self, df, separator="_"):
        """
        Joins the values of a Pandas MultiIndex into one string per row.

        Args:
        df: A pandas DataFrame with a MultiIndex.separator: A string separator used to join MultiIndex values.

        Returns:
        A pandas Series with the joined strings for each row.
        """
        # Ensure the DataFrame has a MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("The DataFrame does not have a MultiIndex.")

        # Join the MultiIndex levels with the specified separator
        return df.index.to_series().apply(lambda x: separator.join(map(str, x)))

    def transpose_data(self):
        # Transpose the dataframe, so loci are rows and samples are columns
        self.data = self.data.T.reset_index()

        # Rename columns to reflect that the first column is now 'sample'
        self.data.rename(columns={"index": self.header_key}, inplace=True)

        # Extract SNP IDs from the column names
        self.snp_ids = self.data.columns[1:].tolist()
        self.snp_ids = ["_".join(x) for x in self.snp_ids]

        # Extract sample IDs which are now the column names, starting from the first sample column
        self.sample_ids = self.data[self.header_key].tolist()
        self.sample_ids = [x for x in self.sample_ids if x not in ["REF", "ALT"]]

    def create_vcf_header(self):
        # VCF header lines start with '##' and the column header line starts with '#'

        self.sample_ids = [
            x.split("_")[-1] for x in self.sample_ids if x.startswith("GTseek")
        ]
        sample_ids = "\t".join(self.sample_ids)

        self.header = [
            "##fileformat=VCFv4.2",
            "##source=GTseqToVCFConverter",
            "##reference=GenomeRef",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + sample_ids.strip("\n"),
        ]

    def format_genotype_data(self):
        """Formats the genotype data for each SNP according to VCF specifications."""
        self.data = self.data.T
        self.data = self.data.iloc[1:, :]
        self.data["ID"] = self.join_multiindex(self.data)
        self.data = self.data.reset_index()
        self.data.columns = ["CHROM", "POS"] + self.sample_ids + ["REF", "ALT", "ID"]
        self.data = self.data.set_index("ID")

        # Ensure SNP IDs are sorted properly for consistent output
        self.snp_ids = sorted(
            self.snp_ids, key=lambda x: (x.split("_")[0], int(x.split("_")[1]))
        )

        # Replace genotype strings with VCF format genotypes
        for snp in self.snp_ids:
            ref_allele = self.data.at[snp, "REF"]
            alt_allele = self.data.at[snp, "ALT"]

            for sample in self.sample_ids:
                genotype = self.data.at[snp, sample]

                # Initialize a list to hold the formatted genotype indices
                formatted_genotype = []

                # Iterate over each character in the genotype string
                for allele in genotype:
                    if allele == ref_allele:
                        formatted_genotype.append("0")
                    elif allele == alt_allele or allele in "ATCG":
                        formatted_genotype.append("1")
                    else:
                        formatted_genotype.append(
                            "."
                        )  # For missing or unrecognized alleles

                # Join the allele indices with '/' to form the VCF genotype
                vcf_genotype = "/".join(formatted_genotype)
                self.data.at[snp, sample] = vcf_genotype

        self.data = self.data.copy()
        self.data["ID"] = self.data.index.copy()

        # Format the data for VCF output
        for snp in self.snp_ids:
            chrom = self.data.at[snp, "CHROM"]
            pos = self.data.at[snp, "POS"]
            ref = self.data.at[snp, "REF"]
            alt = self.data.at[snp, "ALT"]
            loc_id = self.data.at[snp, "ID"]

            # Construct the VCF line for the SNP
            vcf_line = [chrom, pos, loc_id, ref, alt, ".", "PASS", ".", "GT"]

            # Append the genotype data for each sample
            genotypes = [self.data.at[snp, sample] for sample in self.sample_ids]
            vcf_line.extend(genotypes)

            # Join the line into a string and append to the VCF data
            self.vcf_data.append("\t".join(vcf_line))

    def write_vcf(self, output_filename):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        with open(output_filename, "w") as vcf_file:
            # Write the header lines
            for line in self.header:
                vcf_file.write(line + "\n")
            # Write the SNP data lines
            for line in self.vcf_data:
                vcf_file.write(line + "\n")

    def subset_by_locus_ids(self):
        """
        Subsets the data by the locus IDs which are the joined CHROM and POS columns.

        Raises:
            ValueError: If snp_ids have not been set before this method is called.

        Returns:
            pandas.DataFrame: A subset of the original DataFrame containing only the rows with index matching the locus IDs in self.snp_ids.
        """

        # Ensure that the snp_ids attribute has been populated
        if self.snp_ids is None:
            raise ValueError("snp_ids must be set before calling subset_by_locus_ids.")

        # Create a MultiIndex from the CHROM and POS columns if it's not already done
        if not isinstance(self.data.index, pd.MultiIndex):
            self.data.set_index(["CHROM", "POS"], inplace=True)

        # Find the intersection of the data's index with the snp_ids
        locus_ids = set(self.join_multiindex(self.data, separator="_"))
        subset_ids = locus_ids.intersection(self.snp_ids)

        # Subset the DataFrame based on the matching locus IDs
        subset_data = self.data.loc[subset_data.index.isin(subset_ids)]

        return subset_data

    def subset_vcf_by_locus_ids(
        self, vcf_path, output_path, output_filename="phase6_gtseq_subset.vcf"
    ):
        """
        Subsets the VCF file by the locus IDs which are the joined CHROM and POS columns.

        Args:
            vcf_path (str): Path to the input VCF file.
            output_path (str): Directory path where the subsetted VCF file will be written.
            output_filename (str, optional): Name of the subsetted VCF file.Defaults to "phase6_gtseq_subset.vcf".

        Returns:
            None
        """
        # Open the VCF file using pysam
        vcf_in = pysam.VariantFile(vcf_path, "r")

        Path(output_path).mkdir(exist_ok=True, parents=True)
        pth = Path(output_path, output_filename)

        # Create a new VCF file for the output
        vcf_out = pysam.VariantFile(pth, "w", header=vcf_in.header)

        # Read through the VCF file and write only the records with locus IDs in self.snp_ids
        for record in vcf_in:
            locus_id = f"{record.chrom}_{record.pos}"
            if locus_id in self.snp_ids:
                vcf_out.write(record)

        # Close the VCF files
        vcf_in.close()
        vcf_out.close()

    def read_vcf(self, filepath):
        """
        Open a VCF file using pysam and return the VariantFile object.

        Args:
            filepath (str): The file path to the VCF file.

        Returns:
            pysam.VariantFile: VariantFile object for the VCF.
        """
        return pysam.VariantFile(filepath)

    def calculate_missing_percentage_for_samples(self, vcf):
        """
        Calculate the percentage of missing genotype data for each sample in a VCF file.

        Args:
            vcf (pysam.VariantFile): VCF file object.

        Returns:
            dict: A dictionary with sample names as keys and the percentage of missing data as values.
        """
        sample_missing_counts = {sample: 0 for sample in vcf.header.samples}
        total_loci = 0

        for record in vcf:
            total_loci += 1
            for sample in vcf.header.samples:
                if record.samples[sample]["GT"] == (None, None):  # Missing genotype
                    sample_missing_counts[sample] += 1

        return {
            sample: (missing_count / total_loci) * 100
            for sample, missing_count in sample_missing_counts.items()
        }

    def calculate_missing_percentage_for_loci(self, vcf_path):
        """
        Calculate the percentage of missing genotype data for each locus in a VCF file.

        Args:
            vcf_path (str): The file path to the VCF file.

        Returns:
            dict: A dictionary with locus identifiers (CHROM_POS) as keys and the percentage of missing data as values.
        """
        vcf = pysam.VariantFile(vcf_path)
        sample_count = len(vcf.header.samples)
        loci_missing_counts = {}

        for record in vcf:
            missing_count = sum(
                1
                for sample in record.samples
                if record.samples[sample]["GT"] == (None, None)
            )
            loci_missing_counts[f"{record.chrom}_{record.pos}"] = (
                missing_count / sample_count
            ) * 100

        return loci_missing_counts

    def index_vcf(self, filepath):
        """
        Index a VCF file using pysam.

        Args:
            filepath (str): The file path to the VCF file to be indexed.
        """
        pysam.tabix_index(str(filepath), preset="vcf", force=True)
        print(f"Indexed VCF file: {filepath}")

    def merge_vcf_files_with_missing_data_handling(
        self, vcf1_path, vcf2_path, output_path
    ):
        """
        Merge two VCF files by choosing for each locus the genotypes from the file with the least missing data. If a sample is only present in one file, its genotype is still included.

        Args:
            vcf1_path (str): Path to the first VCF file.
            vcf2_path (str): Path to the second VCF file.
            output_path (pathlib.Path): Path to the output VCF file.
        """

        output_dir = output_path.parent
        output_path = str(output_path)

        # Calculate missing data percentages for each locus in both VCFs
        vcf1_missing = self.calculate_missing_percentage_for_loci(vcf1_path)
        vcf2_missing = self.calculate_missing_percentage_for_loci(vcf2_path)

        locus_details = self.calculate_overall_concordance(vcf1_path, vcf2_path)
        self.plot_overall_discordance(locus_details, output_dir)

        # Open VCF files
        vcf1 = pysam.VariantFile(vcf1_path)
        vcf2 = pysam.VariantFile(vcf2_path)

        # Prepare output VCF with header from vcf1, including all samples from both files
        combined_samples = sorted(set(vcf1.header.samples) | set(vcf2.header.samples))
        header = vcf1.header.copy()
        for sample in combined_samples:
            if sample not in vcf1.header.samples:
                header.add_sample(sample)
        output_vcf = pysam.VariantFile(output_path, "w", header=header)

        # Cache vcf2 records for efficiency
        vcf2_records = {f"{rec.chrom}_{rec.pos}": rec for rec in vcf2.fetch()}

        for record1 in vcf1.fetch():
            locus_key = f"{record1.chrom}_{record1.pos}"
            record2 = vcf2_records.get(locus_key, None)

            new_record = output_vcf.new_record()
            new_record.chrom = record1.chrom
            new_record.pos = record1.pos
            new_record.ref = record1.ref
            new_record.id = record1.id
            new_record.alts = record1.alts
            new_record.qual = record1.qual

            # Setting FILTER and INFO fields from the chosen record (prioritize record1 if equal or no record2)
            if record2 and vcf2_missing.get(locus_key, 0) < vcf1_missing.get(
                locus_key, float("inf")
            ):
                for filter_key in record2.filter.keys():
                    new_record.filter.add(filter_key)
                for key, value in record2.info.items():
                    new_record.info[key] = value
            else:
                for filter_key in record1.filter.keys():
                    new_record.filter.add(filter_key)
                for key, value in record1.info.items():
                    new_record.info[key] = value

            for sample in combined_samples:
                gt1 = (None, None)
                if sample in record1.header.samples:
                    gt1 = (
                        record1.samples[sample]["GT"]
                        if record1.samples[sample]["GT"] != (None, None)
                        else (None, None)
                    )

                gt2 = (None, None)
                if record2 and sample in record2.header.samples:
                    gt2 = (
                        record2.samples[sample]["GT"]
                        if record2.samples[sample]["GT"] != (None, None)
                        else (None, None)
                    )

                # Determine which genotype to use based on missing data
                if not record2 or (
                    gt1 != (None, None)
                    and vcf1_missing.get(locus_key, 0)
                    <= vcf2_missing.get(locus_key, float("inf"))
                ):
                    chosen_gt = gt1
                else:
                    chosen_gt = gt2 if gt2 != (None, None) else gt1

                # Apply the chosen genotype
                if chosen_gt != (
                    None,
                    None,
                ):  # Avoid setting explicitly missing genotypes if not necessary
                    new_record.samples[sample]["GT"] = chosen_gt
                else:
                    # Handle explicitly missing data if needed (e.g., set to './.' or leave as default missing)
                    pass

            # Write the new record after setting all sample genotypes
            output_vcf.write(new_record)

        output_vcf.close()

    def find_shared_loci(self, vcf1_path, vcf2_path):
        vcf1 = pysam.VariantFile(vcf1_path)
        vcf2 = pysam.VariantFile(vcf2_path)

        # Extract loci from both files
        vcf1_loci = {f"{record.chrom}-{record.pos}" for record in vcf1.fetch()}
        vcf2_loci = {f"{record.chrom}-{record.pos}" for record in vcf2.fetch()}

        # Determine shared loci
        shared_loci = vcf1_loci.intersection(vcf2_loci)

        return [x.split("-") for x in shared_loci]

    def calculate_overall_concordance(self, vcf1_path, vcf2_path):
        vcf1 = pysam.VariantFile(vcf1_path)
        vcf2 = pysam.VariantFile(vcf2_path)

        # Assuming find_shared_loci() identifies shared loci accurately
        shared_loci = self.find_shared_loci(vcf1_path, vcf2_path)

        locus_details = []

        for chrom, pos in shared_loci:
            pos = int(pos)
            record1 = next(vcf1.fetch(chrom, pos - 1, pos), None)
            record2 = next(vcf2.fetch(chrom, pos - 1, pos), None)

            if not record1 or not record2:
                continue

            # Initialize counts
            concordant_ref = concordant_alt = discordant = 0

            # Compare REF and ALT alleles
            if record1.ref == record2.ref:
                concordant_ref += 1
            else:
                discordant += 1

            if record1.alts == record2.alts:
                concordant_alt += 1
            else:
                discordant += 1

            locus_details.append(
                {
                    "chrom": chrom,
                    "pos": pos,
                    "concordant_ref": concordant_ref,
                    "concordant_alt": concordant_alt,
                    "discordant": discordant,
                }
            )

        return locus_details

    def plot_overall_discordance(self, locus_details, output_dir):
        plt.figure(figsize=(12, 6))

        # Dictionary to hold counts for each discordance category
        discordance_counts = {
            "No Discordance": 0,
            "One Allele Discordant": 0,
            "Both Alleles Discordant": 0,
        }

        # Analyze each locus detail to categorize discordance
        for detail in locus_details:
            if detail["discordant"] == 0:
                discordance_counts["No Discordance"] += 1
            elif detail["discordant"] == 1:
                discordance_counts["One Allele Discordant"] += 1
            else:  # Assuming 'discordant' > 1 means both alleles are discordant
                discordance_counts["Both Alleles Discordant"] += 1

        # Prepare data for plotting
        data_to_plot = pd.DataFrame(
            list(discordance_counts.items()), columns=["Discordance Category", "Count"]
        )

        # Plotting
        sns.barplot(
            x="Discordance Category",
            y="Count",
            hue="Discordance Category",
            data=data_to_plot,
            palette="Set2",
            legend=False,
            hue_order=[
                "No Discordance",
                "One Allele Discordant",
                "Both Alleles Discordant",
            ],
        )
        plt.title("Counts of Allelic Discordance Rates", fontsize=18)
        plt.xlabel("Discordance Category", fontsize=18)
        plt.ylabel("Number of Loci", fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # Save the plot
        pth = Path(output_dir, "overall_discordance.png")
        plt.savefig(pth, dpi=300, bbox_inches="tight")

    def replace_missing_genotypes(self, input_vcf_path, output_vcf_path):
        # Check if the input file is gzip-compressed based on its extension.
        if input_vcf_path.endswith(".gz"):
            file_open = gzip.open
        else:
            file_open = open

        # Use the determined open function to read the input file appropriately.
        with file_open(input_vcf_path, "rt") as original, open(
            output_vcf_path, "w"
        ) as updated:
            for line in original:
                if not line.startswith("#"):
                    # Process non-header lines.
                    entries = line.strip().split("\t")
                    format_idx = (
                        entries[8].split(":").index("GT")
                        if "GT" in entries[8]
                        else None
                    )

                    # Only proceed if the GT format is found.
                    if format_idx is not None:
                        for i in range(9, len(entries)):
                            sample_data = entries[i].split(":")
                            # Replace missing genotype values with './.'.
                            if sample_data[format_idx] == ".":
                                sample_data[format_idx] = "./."
                            entries[i] = ":".join(sample_data)
                    updated.write("\t".join(entries) + "\n")
                else:
                    # Directly write header lines.
                    updated.write(line)


def get_args():
    """
    Parses command line arguments for converting GTseq data to VCF format,merging with ddRADseq data, and generating discordance visualizations.

    Returns:
        Namespace: An argparse.Namespace object containing all the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="""This tool converts GTseq data into VCF format, merges it with ddRADseq VCF file, and handles missing data. It simplifies the bioinformatics pipeline for genomic researchers and provides a visualization of discordance for each of the shared loci.""",
        epilog="""Example usage:
        python script.py --gtseq mydata.csv --radseq radseqdata.vcf --output_dir ./processed
        
        This command processes 'mydata.csv', merges it with 'radseqdata.vcf', and outputs a merged VCF along with discordance visualization to the './processed' directory.
        
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gtseq",
        type=str,
        required=True,
        help="Path to the input GTseq CSV file. The file should contain genotyping data\n"
        "in CSV format, obtained from GTseq genotyping assays.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the output files will be stored. This includes\n"
        "the merged VCF file and any discordance visualizations. The directory\n"
        "will be created if it does not exist.",
    )
    parser.add_argument(
        "--radseq",
        type=str,
        required=True,
        help="Path to the input ddRADseq VCF file. This file should contain variant\n"
        "calling data from ddRADseq sequencing assays, formatted as a VCF file.",
    )
    parser.add_argument(
        "--str2drop",
        type=str,
        nargs="*",
        default=[],
        help="Optional. A list of strings to be excluded from the column names\n"
        "in the GTseq CSV file. Useful for removing unwanted data or metadata\n"
        "columns from the analysis.",
    )
    return parser.parse_args()


# Ensure this script is the main program and not being imported
if __name__ == "__main__":
    args = get_args()
    # The rest of the script would follow here, utilizing the parsed arguments


if __name__ == "__main__":
    args = get_args()

    # Directory and file setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output VCF paths are now explicitly defined to avoid double ".gz" appending
    gtseq_output_vcf_path = output_dir / "gtseq_output.vcf"
    subset_vcf_path = output_dir / "radseq_subset.vcf"  # Explicitly include .gz here
    merged_vcf_path = output_dir / "merged_gtseq_radseq.tmp.vcf"

    # Initialize and process GTseq data
    converter = GTseqToVCF(filepath=args.gtseq, str2drop=args.str2drop)
    converter.load_data()
    converter.parse_sample_column()
    converter.calculate_ref_alt_alleles()
    converter.transpose_data()
    converter.create_vcf_header()
    converter.format_genotype_data()

    # Write the GTseq VCF and index it
    converter.write_vcf(gtseq_output_vcf_path)
    converter.index_vcf(gtseq_output_vcf_path)

    # Subset RADseq VCF based on GTseq loci, write it, and index it
    converter.subset_vcf_by_locus_ids(
        args.radseq,
        str(output_dir),
        output_filename=subset_vcf_path.name,  # Use the .name attribute to get the filename only
    )
    converter.index_vcf(subset_vcf_path)

    # Merge the GTseq VCF with the subsetted RADseq VCF
    converter.merge_vcf_files_with_missing_data_handling(
        str(subset_vcf_path) + ".gz",
        str(gtseq_output_vcf_path) + ".gz",
        merged_vcf_path.with_suffix(".vcf.gz"),
    )

    # Create the new filename by removing '.tmp' from the stem, then adding back the rest of the desired suffix
    new_vcf_filename = merged_vcf_path.stem.replace(".tmp", "") + ".vcf.gz"

    # Use the .with_name() method to get the new path with the corrected filename
    new_vcf_path = merged_vcf_path.with_name(new_vcf_filename)

    converter.replace_missing_genotypes(
        str(merged_vcf_path.with_suffix(".vcf.gz")), str(new_vcf_path)
    )

    merged_vcf_path.with_suffix(".vcf.gz").unlink()
