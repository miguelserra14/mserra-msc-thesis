import os
import torch
import pydot
import shutil
from torch_geometric.data import Data
from tqdm import tqdm

# --- CONFIGURATION ---
# path to embeddings
EMBEDDINGS_FOLDER = "dataset_final_embeddings" 

# path to original graphs
DOTS_ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# where to save final dataset
FINAL_FOLDER = "train_dataset"

GRAPH_TYPES = ["ast", "cfg", "pdg"]

print("Fusion graphs + embeddings")

def process_fusion(intermediate_pt_path, final_output_path):
    try:
        # Load intermediate file
        intermediate_data = torch.load(intermediate_pt_path)
        
        # Retrieve saved information
        node_embeddings_dict = intermediate_data['embeddings']
        label = intermediate_data['label']
        # Relative path to dot
        relative_dot_path = intermediate_data['dot_path'] 
        
        # Reconstruct absolute DOT path
        full_dot_path = os.path.join(DOTS_ROOT_FOLDER, relative_dot_path)
        
        if not os.path.exists(full_dot_path):
            # Try fallback 
            return False

        # Read Graph Structure (.dot)
        graphs = pydot.graph_from_dot_file(full_dot_path)
        if not graphs: return False
        graph = graphs[0]
        
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        # If graph is too small, ignore (noise)
        if len(nodes) < 2: return False

        # Align Data (Map IDs -> Indices 0,1,2...)
        id_to_index = {}
        x_list = []
        
        # Order here is critical: must be same order as nodes in graph
        for index, node in enumerate(nodes):
            node_id = node.get_name().strip('"')
            id_to_index[node_id] = index
            
            # Get corresponding embedding
            if node_id in node_embeddings_dict:
                emb = node_embeddings_dict[node_id]
            else:
                # If fails (empty node), use zeros
                emb = torch.zeros(768)
            
            x_list.append(emb)

        # Create Feature Tensor (X)
        x = torch.stack(x_list)

        # Create Edge Tensor (Edge Index)
        source_indices = []
        target_indices = []
        
        for edge in edges:
            src = edge.get_source().strip('"')
            dst = edge.get_destination().strip('"')
            
            if src in id_to_index and dst in id_to_index:
                source_indices.append(id_to_index[src])
                target_indices.append(id_to_index[dst])
        
        # PyTorch Geometric format [2, num_edges]
        edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

        # Create Final Data Object
        # This is the exact object that DGCNN will consume
        data = Data(x=x, edge_index=edge_index, y=label)
        
        # Save
        torch.save(data, final_output_path)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

total_success = 0

for graph_type in GRAPH_TYPES:
    source_folder = os.path.join(EMBEDDINGS_FOLDER, graph_type)
    dest_folder = os.path.join(FINAL_FOLDER, graph_type)
    
    if not os.path.exists(source_folder):
        print(f" Folder not found: {source_folder}")
        continue
        
    os.makedirs(dest_folder, exist_ok=True)
    
    files = [f for f in os.listdir(source_folder) if f.endswith(".pt")]
    print(f"\n Processing {len(files)} files of {graph_type.upper()}...")
    
    type_success = 0
    for file in tqdm(files):
        source_path = os.path.join(source_folder, file)
        dest_path = os.path.join(dest_folder, file)
        
        if process_fusion(source_path, dest_path):
            type_success += 1
            total_success += 1
            
    print(f" {graph_type.upper()}: {type_success} graphs ready.")

print(f"\n FUSION COMPLETE. {total_success} files ready in '{FINAL_FOLDER}'.")