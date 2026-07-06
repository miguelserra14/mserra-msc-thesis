import pandas as pd
import re
import csv
import json
import ast
import sys

# File paths
cfg_dataset_path = "C:\\Users\\luka3\\Desktop\\UC\\MSI\\Tese\\code\\uc-msc-luisv-cfg-dataset\\datasets\\cfg-dataset-linux-v0.5.csv"
linux_functions_path = "C:\\Users\\luka3\\Desktop\\UC\\MSI\\Tese\\code\\uc-msc-luisv-graph_extractor\\function-data\\linux-functions.csv"
extractor_output_path = "output\cfg-dataset-linux.csv"

# Load datasets
cfg_dataset_df = pd.read_csv(cfg_dataset_path, delimiter=";")
linux_functions_df = pd.read_csv(linux_functions_path)
csv.field_size_limit(10**7)

# Extract commit hashes and file paths from CFG dataset
def extract_commit_and_filepath(cfg_path):
    match = re.search(r'output-cfg-([a-f0-9]+)/(.+?)/\d+-cfg\.dot$', str(cfg_path))
    if match:
        return match.group(1), match.group(2).replace('---', '/')
    return None, None

cfg_dataset_df["Commit Hash"], cfg_dataset_df["Linux File Path"] = zip(
    *cfg_dataset_df["cfg_filepath"].apply(extract_commit_and_filepath)
)

# Function to filter functions based on user selection
def select_code_units(vulnerable=True, cfg_size=1, num_samples=10):
    """
    Selects a specified number of vulnerable or non-vulnerable functions
    that generate CFGs of the given size.
    
    :param vulnerable: Boolean - True for vulnerable functions, False for non-vulnerable.
    :param cfg_size: int - Size of the CFG (1 for single-node, >1 for multi-node).
    :param num_samples: int - Number of functions to retrieve.
    :return: DataFrame - Filtered functions in the same format as linux-functions.csv.
    """

    # Filter CFG dataset by size
    if cfg_size == 1:
        cfg_filtered = cfg_dataset_df[cfg_dataset_df["size"] == 1]
    else:
        cfg_filtered = cfg_dataset_df[cfg_dataset_df["size"] > 1]

    # Filter Linux functions dataset to only include entries that match the CFG dataset
    filtered_functions = linux_functions_df[
        (linux_functions_df["File Path"].isin(cfg_filtered["Linux File Path"])) &
        (linux_functions_df["Vulnerable Commit Hash"].isin(cfg_filtered["Commit Hash"]))
    ]

    # Select vulnerable or non-vulnerable functions
    if vulnerable:
        selected_functions = filtered_functions[filtered_functions["Vulnerable File Functions"].notna()]
    else:
        selected_functions = filtered_functions[filtered_functions["Vulnerable File Functions"].isna()]

    # Sample the requested number of functions
    selected_subset = selected_functions.sample(n=min(num_samples, len(selected_functions)), random_state=42)

    return selected_subset

def create_subset_linux_functions(num_entries=10, output_path="subset_linux_functions.csv"):
    """
    Creates a random subset of entries from linux-functions.csv and saves it as a new CSV file.
    
    :param num_entries: Number of entries to select.
    :param output_path: Path to save the subset CSV file.
    :return: The subset DataFrame.
    """
    
    # Sample entries
    subset_df = linux_functions_df.sample(n=min(num_entries, len(linux_functions_df)), random_state=42)
    
    # Save to CSV
    subset_df.to_csv(output_path, index=False)
    
    print(f"Subset of {len(subset_df)} entries saved to {output_path}")
    return subset_df

def count_functions_per_program(csv_file_path):
    """
    Counts the number of functions per program (per row) in the 'Vulnerable File Functions'
    column of the given CSV file.
    
    Parameters:
        csv_file_path (str): Path to the CSV file.
    
    Returns:
        dict: Dictionary mapping the 'File Path' to the number of functions in that row.
    """
    functions_per_program = {}

    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            file_path = row['File Path']
            functions_list_str = row['Vulnerable File Functions']
            
            try:
                functions_list = json.loads(functions_list_str)
            except json.JSONDecodeError:
                # Skip rows that can't be parsed
                continue
            
            # Number of functions for this program
            functions_per_program[file_path] = len(functions_list)

    return functions_per_program

def check_memory_features_in_graphs(csv_path):
    graphs_with_mm_features = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # Skip header

        for row in reader:
            if len(row) != 5:
                continue  # skip malformed rows

            cfg_path, label, size, adj_matrix_str, feat_matrix_str = row
            size = int(size)
            feature_matrix_flat = ast.literal_eval(feat_matrix_str)

            if len(feature_matrix_flat) != size * 19:
                print(f"[WARN] Feature matrix size mismatch in {cfg_path}")
                continue

            found_mm = False
            for i in range(size):
                node_features = feature_matrix_flat[i * 19:(i + 1) * 19]
                mm_features = node_features[-8:]
                if any(f > 0 for f in mm_features):
                    found_mm = True
                    break

            if found_mm:
                graphs_with_mm_features.append(cfg_path)

    return graphs_with_mm_features

def get_largest_graph_info(csv_path):
    max_size = -1
    max_cfg_path = None
    max_line_number = -1

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)  # skip header

        for line_number, row in enumerate(reader, start=2):  # start=2 to account for header
            if len(row) < 3:
                continue
            try:
                size = int(row[2])
                if size > max_size:
                    max_size = size
                    max_cfg_path = row[0]
                    max_line_number = line_number
            except ValueError:
                continue

    print(f"Grafo com maior size:")
    print(f"  Filepath: {max_cfg_path}")
    print(f"  Size: {max_size}")
    print(f"  Linha no CSV: {max_line_number}")
    return max_cfg_path, max_size, max_line_number


# Example usage for function counting
#file_path = "function-data\\subset_linux_functions.csv"
#functions_count = count_functions_per_program(file_path)
#for program, count in functions_count.items():
#     print(f"{program}: {count} functions")

def main():
    path, size, line = get_largest_graph_info(extractor_output_path)
    results = check_memory_features_in_graphs(extractor_output_path)

    print("Graphs with memory management features:")
    for path in results:
        print(path)

if __name__ == "__main__":
    main()