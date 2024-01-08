import csv
import glob
import os


def calculate_individual_runtimes_and_export_csv(directory_pattern, csv_file_path):
    """
    Calculate the runtime for each file based on the modification time difference with the previous file
    and export the results to a CSV file.

    Args:
        directory_pattern (str): The pattern of the directory and files to search for.
        csv_file_path (str): The path of the CSV file to write the results to.

    Returns:
        None: The function writes the output directly to a CSV file.
    """
    # Get all files matching the pattern
    file_list = glob.glob(directory_pattern)
    if len(file_list) < 2:
        raise ValueError("Not enough files to estimate individual runtimes.")

    # Sort files by modification time
    file_list.sort(key=lambda x: os.path.getmtime(x))

    # Calculate the runtime for each file and write to CSV
    with open(csv_file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Function", "Runtime (seconds)"])

        for i in range(1, len(file_list)):
            prev_file = file_list[i - 1]
            current_file = file_list[i]
            runtime = os.path.getmtime(current_file) - os.path.getmtime(prev_file)
            csvwriter.writerow(["locator", runtime])


def calculate_runtime(directory_pattern):
    """
    Calculate the runtime of a script based on the file modification times in a directory.

    Args:
        directory_pattern (str): The pattern of the directory and files to search for.

    Returns:
        float: The estimated total runtime in seconds.
    """
    # Get all files matching the pattern
    file_list = glob.glob(directory_pattern)
    if len(file_list) < 2:
        raise ValueError("Not enough files to estimate runtime.")

    # Sort files by modification time
    file_list.sort(key=lambda x: os.path.getmtime(x))

    # Calculate the total runtime
    total_runtime = 0
    for i in range(1, len(file_list)):
        start_time = os.path.getmtime(file_list[i - 1])
        end_time = os.path.getmtime(file_list[i])
        total_runtime += end_time - start_time

    return total_runtime


# Example usage
directory_pattern = "locator/*_boot*_predlocs.txt"
csv_file_path = "locator_execution_times.csv"
calculate_individual_runtimes_and_export_csv(directory_pattern, csv_file_path)
