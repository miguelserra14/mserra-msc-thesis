from graph_extraction import extract_graph_per_file
import os
import pandas as pd
from version_management import load_commit
from convert_dot_to_json import map_functions_to_graph, map_single_commit_to_graph


base_project_directory = "/home/constanca/tese_workspace"
commit_data_directory = "function-data"
commit_data_mask = "{}-functions.csv"
commit_data = "linux-functions.csv"
filepath = os.path.join(commit_data_directory, commit_data)
print(filepath)
VULNERABLE_COMMIT_HASH = "Vulnerable Commit Hash"
FILE_PATH = "File Path"
SUBSET = True
#GRAPH_TYPE = "ast"

base_output_directory = "/home/constanca/tese_workspace/uc-msc-constancac-graph-extraction/output"


def obtain_commits_files(project):
    # each file of "commit_data_directory" (function-data) is in the GitHub from João Henggeler
    # scripts/output/final/[PROJECT_NAME]/affected-files-[PROJECT_ID]-[PROJECT]-master-branch-[date].csv
    # e.g., "scripts/output/final/mozilla/affected-files-1-mozilla-master-branch-20210401212440.csv"
    filepath = os.path.join(commit_data_directory, commit_data)
    df = pd.read_csv(filepath)
    commits = df[VULNERABLE_COMMIT_HASH].tolist()
    commits = list(set(commits))
    commits_files = {}
    for commit in commits:
        commits_files[commit] = df[df[VULNERABLE_COMMIT_HASH] == commit][FILE_PATH].tolist()
    print(f"Obtained {len(commits_files)} commits with files for project {project}.")
    return commits_files


# Function to create the output directory for the extraction (if it does not exist already)
# In this project, the folder where the raw .dot and .txt files associated, for each function inside the file, is the 'output' folder.
def check_output_directory(base_output_directory, project):
    output_directory = os.path.join(base_output_directory, project)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

# all projects listed, but we are only using linux
def should_run_per_directory(project):
    return project in ["glibc", "gecko-dev", "linux"]

# being called cfg doesn't mean it only works for cfg, it simply was the first extraction Luis tried. (i change the name to 'graph')
def extract_graph_per_commit_file(project, commits_files, graph_type_list):
    repository_path = os.path.join(base_project_directory, project)
    check_output_directory(base_output_directory, project)
    
    for commit in commits_files:
        # Checkout to the commit
        load_commit(repository_path, commit)

        # Extract graph types (creates .dot files)
        for graph_type in graph_type_list:
            extract_graph_per_file(base_output_directory, project,
                                   repository_path, commit, 
                                   commits_files[commit], graph_type)
        
        # Immediately map this commit's functions to CSV (incremental update)
        print(f"[CSV] Updating CSV mappings for commit {commit[:8]}...")
        map_single_commit_to_graph(project, commit, graph_type_list, filepath)

def summarize_extraction_status(project, graph_type_list):

    filepath = os.path.join(commit_data_directory, commit_data)
    df = pd.read_csv(filepath)

    # unique commit hashes from column "Vulnerable Commit Hash"
    unique_commits = set(df[VULNERABLE_COMMIT_HASH].dropna().unique())
    print(f"\n[INFO] Total unique commits in dataset: {len(unique_commits)}")

    project_output_path = os.path.join(base_output_directory, project)
    os.makedirs(project_output_path, exist_ok=True)
    results = {}

    for graph_type in graph_type_list:
        # search for directories of the type output-ast-* or others (pdg, cfg)
        extracted_commits = {
            folder.split(f"output-{graph_type}-")[-1]
            for folder in os.listdir(project_output_path)
            if folder.startswith(f"output-{graph_type}-")
        }

        missing_commits = unique_commits - extracted_commits

        # To see in command line what is being extracted
        print(f"\n[{graph_type.upper()}]")
        print(f"✓ Extracted: {len(extracted_commits)}")
        print(f"✗ Missing:   {len(missing_commits)}")

        results[graph_type] = {
            "extracted": len(extracted_commits),
            "missing": len(missing_commits),
            "done_commits": list(extracted_commits),
            "missing_commits": list(missing_commits),
        }

    return results


def main():
    projects = ["linux"]
    
    graph_type_list = ["cfg", "ast", "pdg"] 
    
    for project in projects:
        print(f"\n=== Beginning project: {project} ===")
        
        # Verifiy the actual state of the extraction
        summarize_extraction_status(project, graph_type_list)
        
        # Obtain files from linux
        commits_files = obtain_commits_files(project)
        
        # Extract graphs with joern (.dot files)
        print("\nExtracting graphs using Joern")
        extract_graph_per_commit_file(project, commits_files, graph_type_list)
        
        # Creating the artifacts (clean .txt files) and mapping csvs (output-data)
        print("\nCreating .txt files and mappings")
        
        # Iterating for each type of graph
        for graph_type in graph_type_list:
            print(f"\n   >>> Process type: {graph_type.upper()} <<<")
            
            # Ensure CSV completeness (syncs CSVs with all extracted .dot files)
            print("\nFinalizing CSV mappings")
            map_functions_to_graph(project, graph_type, csv_filename=filepath)
            

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
