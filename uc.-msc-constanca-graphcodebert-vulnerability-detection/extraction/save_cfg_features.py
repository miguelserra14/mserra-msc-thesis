import os

def write_cfgs_to_file(project, dataset_samples, graph_type="cfg"):
    graph_dataset_filepath = check_output_file_exists(project,graph_type)
    with open(graph_dataset_filepath, 'a') as output_file:
        for sample in dataset_samples:
            # cfg_filepath,label,size,adjacency_matrix,feature_matrix
            output_file.write("{};{};{};{};{}\n".format(sample[0], sample[1], sample[2], sample[3], sample[4]))
    dataset_samples.clear()


def check_output_file_exists(project,graph_type="cfg"):
    graph_dataset_directory = "./output/"
    graph_dataset_filepath = graph_dataset_directory + f"{graph_type}-dataset-{project}.csv"
    #f"{graph_type}-dataset-{project}.csv"
    if not os.path.exists(graph_dataset_filepath):
        with open(graph_dataset_filepath, 'w') as output_file:
            output_file.write("cfg_filepath;label;size;adjacency_matrix;feature_matrix\n")
    return graph_dataset_filepath
