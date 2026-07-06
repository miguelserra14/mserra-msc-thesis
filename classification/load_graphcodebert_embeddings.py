"""
Load GraphCodeBERT embeddings and convert to DGL format.
Replaces hand-crafted feature matrices with 768-dimensional GraphCodeBERT embeddings.
"""
import os
import pandas as pd
import torch
import dgl
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_embedding_file(file_path):
    """
    Load a single GraphCodeBERT embedding file.
    
    Returns:
        dict with keys: 'embeddings', 'edge_index', 'label', 'function_name', etc.
    """
    try:
        data_dict = torch.load(file_path, map_location='cpu')
        return data_dict
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_dgl_graph_from_embedding(data_dict):
    """
    Convert GraphCodeBERT embedding dict to DGL graph.
    
    Args:
        data_dict: Dict with 'embeddings' (dict of node_id -> tensor[768]),
                   'edge_index' (tensor[2, num_edges]), 'label' (int)
    
    Returns:
        DGL graph with node features stored in ndata['features']
    """
    try:
        # Extract embeddings dictionary and convert to matrix
        embeddings_dict = data_dict['embeddings']
        node_ids = sorted(embeddings_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
        
        # Stack embeddings into matrix [num_nodes, 768] by node id
        node_features = torch.stack([embeddings_dict[node_id] for node_id in node_ids])
        
        # Get edge_index [2, num_edges]
        edge_index = data_dict.get('edge_index', torch.empty((2, 0), dtype=torch.long))
        
        # Create DGL graph from edge list
        if edge_index.numel() > 0:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            g = dgl.graph((src, dst), num_nodes=len(node_ids))
        else:
            # Empty graph (no edges)
            g = dgl.graph(([], []), num_nodes=len(node_ids))
        
        # Store node features
        g.ndata['features'] = node_features.float()
        
        return g
        
    except Exception as e:
        print(f"Error creating DGL graph: {e}")
        return None


def load_embeddings_from_directory(embedding_dir, graph_type='cfg', limit=None):
    """
    Load all GraphCodeBERT embeddings from a directory.
    
    Args:
        embedding_dir: Path to dataset_final_embeddings directory
        graph_type: One of 'ast', 'cfg', 'pdg'
        limit: Optional limit on number of files to load (for testing)
    
    Returns:
        DataFrame with columns ['graphs', 'label'] compatible with detect_vulnerabilities.py
    """
    type_dir = os.path.join(embedding_dir, graph_type)
    
    if not os.path.exists(type_dir):
        raise FileNotFoundError(f"Directory not found: {type_dir}")
    
    # Get all .pt files
    pt_files = [f for f in os.listdir(type_dir) if f.endswith('.pt')]
    
    if limit:
        pt_files = pt_files[:limit]
    
    print(f"Loading {len(pt_files)} {graph_type.upper()} graphs from {type_dir}")
    
    graphs = []
    labels = []
    
    for filename in tqdm(pt_files, desc=f"Loading {graph_type}"):
        file_path = os.path.join(type_dir, filename)
        
        # Load embedding file
        data_dict = load_embedding_file(file_path)
        if data_dict is None:
            continue
        
        # Convert to DGL graph
        g = create_dgl_graph_from_embedding(data_dict)
        if g is None:
            continue
        
        # Extract label
        label = data_dict.get('label', 0)
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        graphs.append(g)
        labels.append(int(label))
    
    print(f"Successfully loaded {len(graphs)} graphs")
    print(f"  Vulnerable: {sum(labels)} | Non-vulnerable: {len(labels) - sum(labels)}")
    
    # Create DataFrame in format expected by detect_vulnerabilities.py
    df = pd.DataFrame({
        'graphs': graphs,
        'label': labels
    })
    
    return df


def save_as_pickle(df, output_path):
    """Save DataFrame as pickle file."""
    df.to_pickle(output_path)
    print(f"Saved to {output_path}")


def main():
    """
    Example usage: Load GraphCodeBERT embeddings and save as pickle.
    """
    # Configuration
    embedding_dir = '/home/constanca/tese_workspace/uc-msc-constancac-graph-extraction/dataset_final_embeddings'
    graph_type = 'cfg'  # or 'ast', 'pdg'
    output_dir = 'datasets'
    
    # Load embeddings
    df = load_embeddings_from_directory(
        embedding_dir=embedding_dir,
        graph_type=graph_type,
        limit=None  # Set to small number for testing, None for all
    )
    
    # Save as pickle
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{graph_type}-graphcodebert-embeddings.pkl')
    save_as_pickle(df, output_path)
    
    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Total graphs: {len(df)}")
    print(f"  Feature dimension: 768 (GraphCodeBERT)")
    
    # Sample graph info
    sample_graph = df.iloc[0]['graphs']
    print(f"\nSample graph:")
    print(f"  Nodes: {sample_graph.num_nodes()}")
    print(f"  Edges: {sample_graph.num_edges()}")
    print(f"  Features shape: {sample_graph.ndata['features'].shape}")


if __name__ == '__main__':
    main()
