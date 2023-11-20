import os
import sys
import pandas as pd
from collections import Counter
import numpy as np
import pysam


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
        Parses the column headers into separate 'CHROM' and 'POS' columns,
        and reorders the DataFrame to have 'CHROM' and 'POS' as the first two columns.
        Validates the 'POS' to be strictly numeric and the genotype columns to contain
        valid allele pairs. Assumes 'Sample' is the index of the DataFrame.

        Args:
            None

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
        Calculates the reference (REF) and alternate (ALT) alleles for each SNP
        across all samples. Assumes the DataFrame has 'CHROM' and 'POS' as a
        MultiIndex for the columns and the rest of the columns are SNP genotypes.
        """

        # Define a helper function to calculate REF and ALT alleles for a series of genotypes
        def get_ref_alt(genotypes):
            # Flatten the genotype pairs and count occurrences, excluding missing data
            allele_counts = Counter(
                "".join(genotypes).replace("--", "").replace("NN", "")
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
        df: A pandas DataFrame with a MultiIndex.
        separator: A string separator used to join MultiIndex values.

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
        sample_ids = "\t".join(self.sample_ids)

        self.header = [
            "##fileformat=VCFv4.2",
            "##source=GTseqToVCFConverter",
            "##reference=GenomeRef\n",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + sample_ids.strip("\n"),
        ]

    def format_genotype_data(self):
        """This method will format the genotype data for each SNP according to VCF specifications."""

        self.data = self.data.T
        self.data = self.data.iloc[1:, :]
        self.data["ID"] = self.join_multiindex(self.data)
        self.data = self.data.reset_index()
        self.data.columns = ["CHROM", "POS"] + self.sample_ids + ["REF", "ALT", "ID"]
        self.data = self.data.set_index("ID")

        # Replace genotype strings with VCF format genotypes
        for snp in self.snp_ids:
            ref_allele = self.data.at[snp, "REF"]
            alt_allele = self.data.at[snp, "ALT"]
            for sample in self.sample_ids:
                genotype = self.data.at[snp, sample]
                alleles = np.array([genotype[i : i + 1] for i in range(len(genotype))])
                alleles[alleles == ref_allele] = "0"
                alleles[alleles == alt_allele] = "1"
                alleles[alleles == "."] = "."
                vcf_genotype = "/".join(alleles)
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
            vcf_line.extend(self.data.loc[snp, self.sample_ids])

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
            output_filename (str, optional): Name of the subsetted VCF file. Defaults to "phase6_gtseq_subset.vcf".

        Returns:
            None
        """
        # Open the VCF file using pysam
        vcf_in = pysam.VariantFile(vcf_path, "r")

        # Create a new VCF file for the output
        vcf_out = pysam.VariantFile(
            os.path.join(output_path, output_filename), "w", header=vcf_in.header
        )

        # Read through the VCF file and write only the records with locus IDs in self.snp_ids
        for record in vcf_in:
            locus_id = f"{record.chrom}_{record.pos}"
            if locus_id in self.snp_ids:
                vcf_out.write(record)

        # Close the VCF files
        vcf_in.close()
        vcf_out.close()


# Example usage of the GTseqToVCF class with the new method
if __name__ == "__main__":
    # Create an instance of the class with the path to your data file
    converter = GTseqToVCF("./WTD_Prod1_Genotypes.csv", str2drop=["Ovi_", "_PRNP_"])

    # Load and preprocess the data
    converter.load_data()
    converter.parse_sample_column()
    converter.calculate_ref_alt_alleles()
    converter.transpose_data()

    # Generate the VCF content
    converter.create_vcf_header()
    converter.format_genotype_data()

    # Write the VCF to a file
    converter.write_vcf("output_path/WTD_Prod1_Genotypes.vcf")

    # Subset the VCF by locus IDs and write to a new file
    converter.subset_vcf_by_locus_ids("phase6.vcf.gz", "output_path")
