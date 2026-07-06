# dot_to_json_graph.py
# http://stackoverflow.com/questions/40262441/how-to-transform-a-dot-graph-to-json-graph

# Packages needed  :
# sudo aptitude install python-networkx python-pygraphviz python-pydot
#
# Syntax :
# python dot_to_json_graph.py graph.dot

from graph_parsing import adjust_file, get_graph_name
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import pandas as pd

base_dot_directory = "output"
commit_data_directory = "function-data"
output_directory = "output-data"
projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
commit_data_mask = "{}-functions.csv"
commit_data="selected_code_units.csv"
filepath = os.path.join(commit_data_directory, commit_data)
VULNERABLE_COMMIT_HASH = "Vulnerable Commit Hash"
FILE_PATH = "File Path"
VULNERABLE_FUNCTIONS = "Vulnerable File Functions"

def convert_dot_to_json(project):
    project_dir = os.path.join(base_dot_directory, project)
    wrong_files = []
    for directory in os.listdir(project_dir)[0:1]:
        print(directory)
        files = os.listdir(os.path.join(project_dir, directory))
        for filename in files:
            try:
                dot_filepath = os.path.join(project_dir, directory, filename)
                adjust_file(dot_filepath)
                dot_graph = nx.nx_pydot.read_dot(dot_filepath)
                print(json.dumps(json_graph.node_link_data(dot_graph)))
            except TypeError:
                wrong_files.append(dot_filepath)
    print("Files with problems ({}): {} ".format(len(wrong_files), wrong_files))


def directory_per_file(project):
    return project in ["glibc", "gecko-dev", "linux"]


def write_output_file(output_filename, rows):
    header = 'commit;filepath;function_name;graph_filepath;vulnerable_label\n'
    output_filepath = os.path.join(output_directory, output_filename)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(output_filepath, 'w') as output_file:
        output_file.write(header)
        for row in rows:
            output_file.write(row)


def map_functions_to_graph(project,graph_type="cfg",csv_filename=filepath):
    # Goals of this function:
    #  1) reads the DF of functions per project
    #  2) iterates over the functions of a file in a commit
    #  3) finds the corresponding file (.dot) of the function
    #  4) creates a CSV file with the output, with the following fields
    #     commit, filepath, function_name, graph_filepath, vulnerable_label
    #filepath = os.path.join(commit_data_directory, commit_data_mask.format(project))
    #filepath = os.path.join(commit_data_directory, commit_data)
    df = pd.read_csv(csv_filename)

    # Filter the samples without vulnerable functions
    df = df[df[VULNERABLE_FUNCTIONS].notnull()]

    csv_rows = []
    for index, row in df.iterrows():
        commit = row[VULNERABLE_COMMIT_HASH]
        filepath = row[FILE_PATH]

        graph_output_directory = os.path.join(base_dot_directory, project, "output-{}-{}".format(graph_type, commit))
        if directory_per_file(project):
            filepath_directory = filepath.replace("/", "---")
            graph_output_directory = os.path.join(graph_output_directory, filepath_directory)

        # Skip if the commit directory doesn't exist (not extracted yet)
        if not os.path.exists(graph_output_directory):
            print(f"[SKIP] Directory not found: {graph_output_directory}")
            continue

        map_function_name_graph_file = {}
        for graph_file in os.listdir(graph_output_directory):
            # Only process .dot files, skip .dot.txt and other files
            if not graph_file.endswith('.dot'):
                continue
            function_name_dot = get_graph_name(graph_output_directory, graph_file)
            map_function_name_graph_file[function_name_dot] = os.path.join(graph_output_directory, graph_file)

        functions = json.loads(row[VULNERABLE_FUNCTIONS])
        for function in functions:
            function_name = function["Name"]
            vulnerable = function["Vulnerable"] == "Yes"

            if function_name in map_function_name_graph_file:
                graph_filepath = map_function_name_graph_file[function_name]
            else:
                # some cases are not found due to the "scope resolution operator" (::)
                # for instance, "nsIndexedToHTML::AsyncConvertData", but in the CSV file
                # only "AsyncConvertData" is present.
                # So we need to find it regardless the scope operator.
                function_in_map = ["::{}".format(function_name) in k for k in map_function_name_graph_file.keys()]

                if True in function_in_map:
                    key_index = function_in_map.index(True)
                    complete_function_name = list(map_function_name_graph_file.keys())[key_index]
                    graph_filepath = map_function_name_graph_file[complete_function_name]
                else:
                    graph_filepath = ""
                    print("Houston, we have a problem...")

            csv_row = "{};{};{};{};{}\n".format(commit, filepath, function_name, graph_filepath, vulnerable)
            print(csv_row)
            csv_rows.append(csv_row)

    write_output_file("functions-{}-{}.csv".format(project,graph_type), csv_rows)


def map_single_commit_to_graph(project, commit, graph_type_list, csv_filename=filepath):
    """
    Map functions for a single commit across all graph types.
    Appends to the CSV files incrementally.
    """
    df = pd.read_csv(csv_filename)
    df = df[df[VULNERABLE_FUNCTIONS].notnull()]
    
    # Filter to only this commit
    commit_df = df[df[VULNERABLE_COMMIT_HASH] == commit]
    
    if commit_df.empty:
        return
    
    # Process each graph type
    for graph_type in graph_type_list:
        csv_rows = []
        
        for index, row in commit_df.iterrows():
            filepath = row[FILE_PATH]
            
            graph_output_directory = os.path.join(base_dot_directory, project, "output-{}-{}".format(graph_type, commit))
            if directory_per_file(project):
                filepath_directory = filepath.replace("/", "---")
                graph_output_directory = os.path.join(graph_output_directory, filepath_directory)
            
            # Skip if the commit directory doesn't exist (not extracted yet)
            if not os.path.exists(graph_output_directory):
                continue
            
            map_function_name_graph_file = {}
            for graph_file in os.listdir(graph_output_directory):
                # Only process .dot files, skip .dot.txt and other files
                if not graph_file.endswith('.dot'):
                    continue
                try:
                    function_name_dot = get_graph_name(graph_output_directory, graph_file)
                    map_function_name_graph_file[function_name_dot] = os.path.join(graph_output_directory, graph_file)
                except:
                    # Skip files that can't be parsed
                    pass
            
            functions = json.loads(row[VULNERABLE_FUNCTIONS])
            for function in functions:
                function_name = function["Name"]
                vulnerable = function["Vulnerable"] == "Yes"
                
                if function_name in map_function_name_graph_file:
                    graph_filepath = map_function_name_graph_file[function_name]
                else:
                    # Handle scope resolution operator
                    function_in_map = ["::{}".format(function_name) in k for k in map_function_name_graph_file.keys()]
                    
                    if True in function_in_map:
                        key_index = function_in_map.index(True)
                        complete_function_name = list(map_function_name_graph_file.keys())[key_index]
                        graph_filepath = map_function_name_graph_file[complete_function_name]
                    else:
                        graph_filepath = ""
                
                csv_row = "{};{};{};{};{}\n".format(commit, filepath, function_name, graph_filepath, vulnerable)
                csv_rows.append(csv_row)
        
        # Append to CSV file (create if doesn't exist)
        if csv_rows:
            output_filepath = os.path.join(output_directory, "functions-{}-{}.csv".format(graph_type, project))
            
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            
            # Write header if file doesn't exist
            if not os.path.exists(output_filepath):
                with open(output_filepath, 'w') as f:
                    f.write('commit;filepath;function_name;graph_filepath;vulnerable_label\n')
            
            # Append rows
            with open(output_filepath, 'a') as f:
                for row in csv_rows:
                    f.write(row)
            
            print(f"[CSV UPDATE] Added {len(csv_rows)} functions for {graph_type} commit {commit[:8]}")


def main():
    for project in projects[3:4]:
        if directory_per_file(project):
            map_functions_to_graph(project)
        else:
            convert_dot_to_json(project)

if __name__ == "__main__":
    main()
