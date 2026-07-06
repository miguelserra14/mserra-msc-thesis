# old luis
import ast
from collections import Counter
import os
import networkx as nx
import numpy as np
import pandas as pd
import html

from graph_extraction_constants import GRAPH_FILE, LABEL
from graph_parsing import read_graph
from feature_identifier_constants import *
from save_cfg_features import write_cfgs_to_file


# This is another interesting link to explain GCN (Graph Convolution Network):
# https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780


# The node types can be of the following types (according to Yan2019):
# 1) Code Sequence:
# -- # Numeric Constants: (assignments in our case)
# -- # Transfer Instructions:
# -- # Call Instructions: METHOD
# -- # Arithmetic Instructions: <operator>.sizeOf, <operator>.cast, <operator>.subtraction, <operator>.division
#                               <operator>.preDecrement, <operator>.minus, <operator>.addition, <operator>.preIncrement
# -- # Compare Instructions: <operator>.equals, <operator>.lessThan, <operator>.logicalNot, <operator>.conditional,
#                            <operator>.logicalOr, <operator>.logicalAnd, <operator>.greaterThan, <operator>.notEquals
# -- # Move Instructions: (not considered)
# -- # Termination Instructions: (return, exit), METHOD_RETURN, RETURN
# -- # Data Declaration Instructions: (variable declaration), <operator>.new, <operator>.assignment
# -- # Total Instructions: (not sure what would be the difference between it and the instructions in the vertex)
# 2) Vertex Structure:
# -- # Offspring (Degree)
# -- # Instructions in the Vertex

# <operator>.indirection, <operator>.addressOf
# NS_LITERAL_CSTRING, NS_SUCCEEDED, NS_ENSURE_TRUE, NS_LITERAL_STRING, NS_ASSERTION, NS_FAILED, NS_ADDREF, UNKNOWN
# pushBuffer.Append, getter_AddRefs, buffer.Append, getter_Copies

# All items have 5 or more occurrences, or they are obviously a reserved node (total of 141 node types)

statements_cfg_folder = "statements-cfg-output"
reduced_cfg_folder = "reduced-cfg-output"

data_directory = "output-data"
file_cfg_data_mask = "functions-{}-{}.csv"
projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]

statement_types = []

other_cases = []
allocation_features = []
deallocation_features = []
unsafe_features = []

#graph_type = "cfg"


def obtain_cfg_data_structures(cfg_filepath, statements_filepath=None):
    A, X, cfg_nx = None, None, None
    if os.path.exists(cfg_filepath):
        graphs = read_graph(cfg_filepath, print_filepath=False)

        if graphs is not None:
            cfg_dot = graphs[0]
            cfg_nx = nx.nx_pydot.from_pydot(cfg_dot)
            # s = Source(cfg_dot, filename="test.gv", format="pdf")
            # s.view()

            # TODO what if we do a topological sorting?
            if len(cfg_nx.nodes()) > 0:
                A, node_order = convert_graph_to_adjacency_matrix(cfg_dot)
                X = obtain_attribute_matrix(cfg_nx, node_order, statements_filepath)
    return A, X, cfg_nx

def obtain_ast_features(ast_nx, node_order, statements_filepath):
    number_features_ast = 8
    X = np.zeros((len(node_order), number_features_ast))

    statement_count, statements_node = obtain_statements(statements_filepath)

    for i, node in enumerate(node_order):
        statements = statements_node.get(node, [])

        # Feature 6: IS_LEAF_NODE (has no children)
        is_leaf = ast_nx.out_degree(node) == 0
        X[i][6] = 1 if is_leaf else 0

        # Feature 7: OUT_DEGREE (structural info)
        X[i][7] = ast_nx.out_degree(node)

        for stmt in statements:
            try:
                stmt_type = stmt[1:stmt.index(",")].upper()
            except Exception:
                continue

            if stmt_type.startswith("LITERAL"):
                X[i][0] = 1
            elif stmt_type.startswith("IDENTIFIER"):
                X[i][1] = 1
            elif stmt_type.startswith("CALL") or "CALL" in stmt_type:
                X[i][2] = 1
            elif stmt_type == "BLOCK":
                X[i][3] = 1
            elif stmt_type == "CONTROL_STRUCTURE":
                X[i][4] = 1
            elif stmt_type == "UNKNOWN":
                X[i][5] = 1

    return X


def obtain_pdg_features(pdg_nx, node_order, statements_filepath):
    number_features_pdg = 7  # DDG_OUT, DDG_IN, IS_CONTROL_DEP, NUM_EDGES_TOTAL, CDG_IN, CDG_OUT, IS_PARAM
    X = np.zeros((len(node_order), number_features_pdg))

    statement_count, statements_node = obtain_statements(statements_filepath)

    for i, node in enumerate(node_order):
        num_ddg_out = 0
        num_ddg_in = 0
        num_cdg_out = 0
        num_cdg_in = 0
        total_edges = 0
        is_control_dependent = 0
        is_param = 0

        # Verifica tipo do statement
        stmts = statements_node.get(node, [])
        for stmt in stmts:
            try:
                stmt_type = stmt[1:stmt.index(",")].upper()
                if stmt_type == "PARAM":
                    is_param = 1
            except Exception:
                continue

        # Arestas de saída
        for succ in pdg_nx.successors(node):
            for k in pdg_nx[node][succ]:
                edge_label = pdg_nx.edges[node, succ, k].get("label", "").strip().strip('"')
                total_edges += 1
                if edge_label.startswith("DDG"):
                    num_ddg_out += 1
                elif edge_label.startswith("CDG"):
                    num_cdg_out += 1
                    is_control_dependent = 1

        # Arestas de entrada
        for pred in pdg_nx.predecessors(node):
            for k in pdg_nx[pred][node]:
                edge_label = pdg_nx.edges[pred, node, k].get("label", "").strip().strip('"')
                total_edges += 1
                if edge_label.startswith("DDG"):
                    num_ddg_in += 1
                elif edge_label.startswith("CDG"):
                    num_cdg_in += 1
                    is_control_dependent = 1

        X[i][0] = num_ddg_out
        X[i][1] = num_ddg_in
        X[i][2] = is_control_dependent
        X[i][3] = total_edges
        X[i][4] = num_cdg_in
        X[i][5] = num_cdg_out
        X[i][6] = is_param

    return X

def obtain_graph_data_structures(graph_type, filepath, statements_filepath=None):
    A, X, nx_graph, node_order = None, None, None, []
    if os.path.exists(filepath):
        graphs = read_graph(filepath, print_filepath=False)

        if graphs is not None:
            dot = graphs[0]
            nx_graph = nx.nx_pydot.from_pydot(dot)

            if len(nx_graph.nodes()) > 0:
                A, node_order = convert_graph_to_adjacency_matrix(dot)

                if graph_type == "cfg":
                    X = obtain_attribute_matrix(nx_graph, node_order, statements_filepath)
                elif graph_type == "ast":
                    X = obtain_ast_features(nx_graph, node_order, statements_filepath)
                elif graph_type == "pdg":
                    X = obtain_pdg_features(nx_graph, node_order, statements_filepath)
                
                return A, X, nx_graph, node_order
                #print(f"Graph: {filepath} - A.shape: {A.shape if A is not None else None}, X.shape: {X.shape if X is not None else None}")
    return A, X, nx_graph, node_order



def convert_graph_to_adjacency_matrix(cfg_dot):
    cfg = nx.nx_pydot.from_pydot(cfg_dot)
    #node_types = obtain_node_types(cfg_dot)

    #print(node_types)

    #adjacency_matrix = nx.adjacency_matrix(cfg)
    #print(type(adjacency_matrix))
    #print(adjacency_matrix)

    #np_matrix = nx.to_numpy_matrix(cfg, nodelist=cfg.nodes())
    #print(np_matrix)
    #print(cfg.nodes())

    different_order_nodes = list(cfg.nodes())
    if len(different_order_nodes) > 1:
        head = different_order_nodes.pop(-2)
        different_order_nodes.insert(0, head)
    #print(different_order_nodes)
    np_matrix = nx.to_numpy_array(cfg, nodelist=different_order_nodes)
    #print(np_matrix)
    return np_matrix, different_order_nodes

def obtain_statements(statements_filepath):
    with open(statements_filepath) as f:
        node_statements = f.readlines()

    statement_count = {}
    statements_node = {}
    for i in range(len(node_statements)):
        node = node_statements[i]
        space_index = node.index(" ")
        node_label = node[0:space_index]
        str_statements = node[space_index + 1:].replace("\n", "")

        # to read the string list, we need to use "ast"
        # str_list  = ['(RETURN,return rv;,return rv;)']
        #statements = ast.literal_eval(str_statements)
        statements = [html.unescape(stmt) for stmt in ast.literal_eval(str_statements)]
        #statement_type = [tp[1:tp.index(",")] for tp in statements]
        statement_type = [html.unescape(tp[1:tp.index(",")]) for tp in statements]
        statement_types.extend(statement_type)

        statement_count[node_label] = len(statements)
        statements_node[node_label] = statements
    return statement_count, statements_node


def get_node_type_id(statement_type):
    if statement_type in statement_type_map:
        return statement_type_map[statement_type]
    else:
        return -1

'''
def obtain_code_sequence_features(number_features_code_sequence, statements):
    X_code_sequence = np.zeros(number_features_code_sequence)
    types = []
    for statement in statements:
        # Ensure we handle statements without a comma
        comma_index = statement.find(",")
        if comma_index != -1:
            statement_type = statement[1:comma_index]
        else:
            statement_type = statement.strip("()")
        types.append(statement_type)
        node_type_id = get_node_type_id(statement_type)
        if node_type_id != -1:
            X_code_sequence[node_type_id] += 1
    return X_code_sequence
'''

def obtain_code_sequence_features(number_features_code_sequence, statements):
    # creates the features based on the code sequence
    X_code_sequence = np.zeros(number_features_code_sequence)
    types = []
    for statement in statements:
        statement_type = statement[1:statement.index(",")]
        types.append(statement_type)
        node_type_id = get_node_type_id(statement_type)
        if node_type_id != -1:
            X_code_sequence[node_type_id] += 1
    #print(types)f
    return X_code_sequence


def count_functions_statement(statement, functions, features_identified):
    count_functions = 0
    for function_name in functions:
        if function_name in statement:
            if statement.startswith("{}(".format(function_name)) or " {}(".format(function_name) in statement:
                count_functions += 1
                #print(statement)
                features_identified.append((function_name, statement))
            else:
                #print("DIFFERENT CASE (function_name, statement): ({}, {})".format(function_name, statement))
                other_cases.append((function_name, statement))
    return count_functions


def count_allocation_functions(statement):
    allocation_functions_list = ["malloc", "calloc", "realloc", "new",
                                 "kmalloc", "vmalloc", "kvmalloc", "vzalloc",
                                 "kcalloc"]
    return count_functions_statement(statement, allocation_functions_list, allocation_features)


def count_deallocation_functions(statement):
    deallocation_functions_list = ["free", "delete", "vfree", "kfree", "kvfree"]
    return count_functions_statement(statement, deallocation_functions_list, deallocation_features)


def count_convert_unsafe_functions(statement):
    convert_unsafe_functions_list = ["atoi", "atol", "atoll", "atof"]
    return count_functions_statement(statement, convert_unsafe_functions_list, unsafe_features)


def count_string_unsafe_functions(statement):
    string_unsafe_functions_list = ["gets", "getpw", "strcat", "strcpy", "sprintf", "vsprintf", "puts", "read_chunk"]
    return count_functions_statement(statement, string_unsafe_functions_list, unsafe_features)


def count_scanf_unsafe_functions(statement):
    scanf_unsafe_functions_list = ["scanf", "fscanf", "sscanf", "vscanf", "vsscanf", "vfscanf"]
    return count_functions_statement(statement, scanf_unsafe_functions_list, unsafe_features)


def count_other_unsafe_functions(statement):
    scanf_other_functions_list = ["realpath", "getopt", "getpass", "streadd", "strecpy", "strtrns"]
    return count_functions_statement(statement, scanf_other_functions_list, unsafe_features)


def count_address_of(statement_type):
    return 1 if statement_type == "<operator>.addressOf" else 0
    # Verificar este caso
    # ---> Nem todos são deste tipo.
    # Ver este caso: /hdd/josep/linux/output-cfg-aa77d26961fa4ecb11fe4209578dcd62ad15819d/security---selinux---hooks.c/233-cfg.dot


def count_pointer_assignment(statement_type, statement):
    if statement_type == "<operator>.assignment" and statement.startswith("*") and ("=" in statement):
        print(statement_type, statement)
        return 1
    return 0


def obtain_feature_mm_count(cfg_statement):
    # TODO parses the statement and check all the types
    list_features = {}
    statement_type = cfg_statement[1:cfg_statement.index(",")]
    statement = cfg_statement[cfg_statement.index(",")+1:-1]
    list_features[ALLOCATION_FUNCTIONS] = count_allocation_functions(statement)
    list_features[DEALLOCATION_FUNCTIONS] = count_deallocation_functions(statement)
    list_features[POINTER_ASSIGNMENT] = count_pointer_assignment(statement_type, statement)
    list_features[MEMORY_ADDRESS_OF] = count_address_of(statement_type)
    list_features[CONVERT_UNSAFE] = count_convert_unsafe_functions(statement)
    list_features[STRING_UNSAFE] = count_string_unsafe_functions(statement)
    list_features[SCANF_UNSAFE] = count_scanf_unsafe_functions(statement)
    list_features[OTHER_UNSAFE] = count_other_unsafe_functions(statement)
    return list_features


def obtain_mm_features(number_features_memory_management, statements):
    # Method to calculate the features related to lack of memory management
    X_memory_management = np.zeros(number_features_memory_management)
    for statement in statements:
        #print(statement)
        mm_features = obtain_feature_mm_count(statement)
        for mm_feature in mm_features:
            X_memory_management[mm_feature] += mm_features[mm_feature]
    return X_memory_management


def obtain_attribute_matrix(cfg_nx, node_order, statements_filepath):
    # This is a matrix with node degrees. Other attributes can be used
    # TODO Normalize this matrix

    number_features_code_sequence = 8
    number_features_vertex_structure = 3
    number_features_memory_management = 8
    total_features = number_features_vertex_structure + \
                     number_features_code_sequence + \
                     number_features_memory_management

    cfg_order = len(node_order)  # number of nodes in the CFG
    if statements_filepath is None:
        statements_count, statements_node = {node_label: 1 for node_label in node_order}, None
    else:
        statements_count, statements_node = obtain_statements(statements_filepath)

    X = np.zeros((cfg_order, total_features))
    for i in range(X.shape[0]):
        X[i][0] = cfg_nx.out_degree(node_order[i])  # outdegree
        X[i][1] = cfg_nx.in_degree(node_order[i])  # indegree
        X[i][2] = statements_count[node_order[i]]  # number of statements
        X[i][number_features_vertex_structure:number_features_vertex_structure+number_features_code_sequence] = \
            obtain_code_sequence_features(number_features_code_sequence, statements_node[node_order[i]])
        X[i][number_features_vertex_structure + number_features_code_sequence:total_features] = \
            obtain_mm_features(number_features_memory_management, statements_node[node_order[i]])

    if statements_filepath is None:
        # removes the column with the number of statements per node
        X = np.delete(X, 2, 1)
    #print(X)
    return X


def get_node_type(label):
    # remove the quotes and parenthesis
    node_type = label[2:-2]
    node_type = node_type[0:node_type.index(",")]
    return node_type


def obtain_node_types(cfg_dot):
    node_types = {}
    for node in cfg_dot.get_nodes():
        node_label = node.get_label()
        #print(node.get_name(), node_label)
        if node_label is not None:
            node_type = get_node_type(node_label)
            if node_type not in node_types:
                node_types[node_type] = 1
            else:
                node_types[node_type] += 1
    #print(node_types)
    return node_types


def obtain_reduced_statement_filepath(project, cfg_filepath, graph_type="cfg"):
    cfg_filepath_parts = cfg_filepath.split("\\")

    cfg_filename = cfg_filepath_parts[-1]
    cfg_name = cfg_filename[:cfg_filename.index(".dot")]
    if project not in ["httpd"]:
        repository_directory = cfg_filepath_parts[-2]
        output_commit_index = -3
        base_directory_max_index = -4
    else:
        repository_directory = ""
        output_commit_index = -2
        base_directory_max_index = -3

    output_commit = cfg_filepath_parts[output_commit_index]
    base_directory = "/".join(cfg_filepath_parts[0:base_directory_max_index])

    cfg_reduced_filepath = os.path.join(base_directory, "{}-{}-reduced".format(project, graph_type),
                                        output_commit, repository_directory, "{}.dot".format(cfg_name))
    statements_filepath = os.path.join(base_directory, "{}-{}-statements".format(project, graph_type),
                                       output_commit, repository_directory, "{}.txt".format(cfg_name))
    return cfg_reduced_filepath, statements_filepath


def fex_read_cfg_file(project,graph_type="cfg"):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project, graph_type))
    df = pd.read_csv(filepath, delimiter=";")
    df = df[df[GRAPH_FILE].notnull()]

    dataset_samples = []
    for index, row in df.iterrows():
        cfg_filepath = row[GRAPH_FILE]
        cfg_reduced_filepath, statements_filepath = obtain_reduced_statement_filepath(project, cfg_filepath)

        A, X, cfg_nx = obtain_cfg_data_structures(cfg_reduced_filepath, statements_filepath)

        if A is not None:
            adjacency_matrix = list(A.ravel())
            adjacency_matrix = [int(a) for a in adjacency_matrix]
            features = list(X.flatten())
            features = [int(f) for f in features]
            dataset_samples.append((cfg_reduced_filepath, row[LABEL], A.shape[0],
                                    adjacency_matrix, features))

        if (index + 1) % 10 == 0:
            write_cfgs_to_file(project, dataset_samples)

    write_cfgs_to_file(project, dataset_samples)

def fex_read_graph_file(project, graph_type):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project, graph_type))
    df = pd.read_csv(filepath, delimiter=";")
    df = df[df[GRAPH_FILE].notnull()]

    dataset_samples = []
    for index, row in df.iterrows():
        graph_filepath = row[GRAPH_FILE]
        graph_reduced_filepath, statements_filepath = obtain_reduced_statement_filepath(project, graph_filepath, graph_type)

        A, X, _, _ = obtain_graph_data_structures(graph_type, graph_reduced_filepath, statements_filepath)

        if A is not None:
            adjacency_matrix = list(A.ravel())
            adjacency_matrix = [int(a) for a in adjacency_matrix]
            features = list(X.flatten())
            features = [int(f) for f in features]
            dataset_samples.append((graph_reduced_filepath, row[LABEL], A.shape[0], adjacency_matrix, features))

        if (index + 1) % 10 == 0:
            write_cfgs_to_file(project, dataset_samples, graph_type)

    write_cfgs_to_file(project, dataset_samples, graph_type)

def debug_all_features_for_specific_graph(graph_filepath, statements_filepath, graph_type="cfg"):
    print(f"\n[DEBUG] Inspecting ALL features for:\n{graph_filepath}")

    A, X, _, node_order = obtain_graph_data_structures(graph_type, graph_filepath, statements_filepath)
    if A is None or X is None:
        print("❌ Could not load graph or features.")
        return

    if graph_type == "cfg":
        number_features_vertex_structure = 3
        number_features_code_sequence = 8
        number_features_memory_management = 8

        # Ranges
        vs_start, vs_end = 0, number_features_vertex_structure
        cs_start, cs_end = vs_end, vs_end + number_features_code_sequence
        mm_start, mm_end = cs_end, cs_end + number_features_memory_management

        # Feature name mappings
        code_seq_feature_names = [
            "NUMERIC_CONSTANT", "TRANSFER_INSTRUCTION", "CALL_INSTRUCTION",
            "ARITHMETIC_INSTRUCTION", "COMPARE_INSTRUCTION", "MOV_INSTRUCTION",
            "TERMINATION_INSTRUCTION", "DATA_DECLARATION_INSTRUCTION"
        ]

        mm_feature_names_by_index = {
            ALLOCATION_FUNCTIONS: "ALLOCATION_FUNCTIONS",
            DEALLOCATION_FUNCTIONS: "DEALLOCATION_FUNCTIONS",
            POINTER_ASSIGNMENT: "POINTER_ASSIGNMENT",
            MEMORY_ADDRESS_OF: "MEMORY_ADDRESS_OF",
            CONVERT_UNSAFE: "CONVERT_UNSAFE",
            STRING_UNSAFE: "STRING_UNSAFE",
            SCANF_UNSAFE: "SCANF_UNSAFE",
            OTHER_UNSAFE: "OTHER_UNSAFE"
        }
        mm_feature_names = [mm_feature_names_by_index[i] for i in sorted(mm_feature_names_by_index.keys())]

        for i in range(X.shape[0]):
            node_id = node_order[i]
            vs_feats = X[i, vs_start:vs_end]
            cs_feats = X[i, cs_start:cs_end]
            mm_feats = X[i, mm_start:mm_end]

            if any(vs_feats) or any(cs_feats) or any(mm_feats):
                print(f"\n[Node {i} | ID {node_id}] Features:")

                if any(vs_feats):
                    print("  Vertex Structure:")
                    print(f"    - Outdegree: {int(vs_feats[0])}")
                    print(f"    - Indegree: {int(vs_feats[1])}")
                    print(f"    - Num Statements: {int(vs_feats[2])}")

                if any(cs_feats):
                    print("  Code Sequence Features:")
                    for j, val in enumerate(cs_feats):
                        if val > 0:
                            print(f"    - {code_seq_feature_names[j]}: {int(val)}")

                if any(mm_feats):
                    print("  Memory Management Features:")
                    for j, val in enumerate(mm_feats):
                        if val > 0:
                            print(f"    - {mm_feature_names[j]}: {int(val)}")

    elif graph_type == "ast":
        ast_feature_names = [
            "IS_LITERAL", "IS_IDENTIFIER", "IS_CALL_NODE",
            "IS_BLOCK", "IS_CONTROL_STRUCTURE", "IS_UNKNOWN",
            "IS_LEAF", "OUT_DEGREE"
        ]

        for i in range(X.shape[0]):
            node_id = node_order[i]
            node_feats = X[i]
            if any(node_feats):
                print(f"\n[Node {i} | ID {node_id}] Features:")
                for j, val in enumerate(node_feats):
                    if val > 0:
                        print(f"    - {ast_feature_names[j]}: {int(val)}")

    elif graph_type == "pdg":
        pdg_feature_names = [
            "DDG_OUT", "DDG_IN", "IS_CONTROL_DEP",
            "TOTAL_EDGES", "CDG_IN", "CDG_OUT", "IS_PARAM"
        ]

        for i in range(X.shape[0]):
            node_id = node_order[i]
            node_feats = X[i]
            if any(node_feats):
                print(f"\n[Node {i} | ID {node_id}] Features:")
                for j, val in enumerate(node_feats):
                    if val > 0:
                        print(f"    - {pdg_feature_names[j]}: {int(val)}")

    else:
        print(f"❌ Graph type '{graph_type}' is not supported.")


def write_output_file(rows, output_filepath):
    rows_list = list(set(rows))
    with open(output_filepath, 'w') as output_file:
        for item in rows_list:
            output_file.write("{},{}\n".format(item[0], item[1]))


def main():
    projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
    project = "linux"  # Change this to the desired project
    graph_type_list = ["ast"]#, "ast", "pdg"]
    graph_type = "pdg"  # Change this to the desired graph type
    commit_hash = "b90c062c65cc8839edfac39778a37a55ca9bda36"  # Example commit hash
    program_name = "arch---x86---kvm---x86.c"  # Example program name
    graph_number = "40"  # Example graph number

    #for graph_type in graph_type_list:
    #        print(f"\n### Processing {project} - {graph_type} ###")
    #        fex_read_graph_file(project,graph_type)                              # from cfg_feature_extraction.py

    cfg_path = "output/linux-{}-reduced/output-{}-{}/{}/{}-{}.dot".format(graph_type, graph_type, commit_hash, program_name, graph_number, graph_type)
    stmts_path = "output/linux-{}-statements/output-{}-{}/{}/{}-{}.txt".format(graph_type, graph_type, commit_hash, program_name, graph_number, graph_type)
    #cfg_path = "output/b90c&b388-reduced/linux-cfg-reduced-b90c&b388reduced/output-cfg-b90c062c65cc8839edfac39778a37a55ca9bda36/arch---x86---kvm---x86.c/84-cfg.dot"
    #stmts_path = "output/b90c&b388-reduced/linux-cfg-statements-b90c&b388reduced/output-cfg-b90c062c65cc8839edfac39778a37a55ca9bda36/arch---x86---kvm---x86.c/84-cfg.txt"

    debug_all_features_for_specific_graph(cfg_path, stmts_path, graph_type)

if __name__ == "__main__":
    main()
