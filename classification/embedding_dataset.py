"""
PyTorch Geometric Dataset for loading graph embeddings.
Converts pre-computed embeddings to PyG Data format for DGCNN training.
"""
import os
import torch
from torch_geometric.data import Dataset, Data
import pandas as pd
from tqdm import tqdm


class EmbeddingDataset(Dataset):
    """
    Load pre-computed graph embeddings and convert to PyG format.
    
    Args:
        root_dir: Path to dataset_final_embeddings folder
        graph_types: List of graph types to include ['ast', 'cfg', 'pdg']
        manifest_csv: Optional path to dataset_manifest.csv for filtering
        transform: Optional transform to apply to Data objects
    """
    
    def __init__(self, root_dir='dataset_final_embeddings',
                 graph_types=['ast', 'cfg', 'pdg'],
                 manifest_csv=None,
                 transform=None):
        self.root_dir = root_dir
        self.graph_types = graph_types if isinstance(graph_types, list) else [graph_types]
        self.transform = transform
        
        # Collect all file paths
        self.file_paths = []
        
        if manifest_csv and os.path.exists(manifest_csv):
            # Load from manifest
            df = pd.read_csv(manifest_csv)
            df = df[df['graph_type'].isin(self.graph_types)]
            self.file_paths = [os.path.join(root_dir, fp) for fp in df['file_path'].values]
        else:
            # Scan directories
            for graph_type in self.graph_types:
                type_dir = os.path.join(root_dir, graph_type) # see if there is a folder inside the root folder for each graph type
                if os.path.exists(type_dir):
                    files = [os.path.join(type_dir, f) for f in os.listdir(type_dir) 
                            if f.endswith('.pt')]
                    self.file_paths.extend(files)
        
        print(f"Loaded {len(self.file_paths)} graphs from {self.graph_types}")
        
        super().__init__(root_dir, transform)
    
    def len(self):
        return len(self.file_paths)
    
    def get(self, idx):
        """
        Load and convert embedding file to PyG Data format.
        
        Returns:
            Data(x, edge_index, y) where:
                x: [num_nodes, 768] node feature matrix (the one we are substituting)
                edge_index: [2, num_edges] edge connectivity
                y: scalar label (0=non-vulnerable, 1=vulnerable)
        """
        file_path = self.file_paths[idx]
        
        try:
            data_dict = torch.load(file_path, map_location='cpu')
            
            # Extract embeddings dictionary and convert to matrix
            embeddings_dict = data_dict['embeddings']
            node_ids = sorted(embeddings_dict.keys(), key=lambda x: int(x) if x.isdigit() else x)
            
            # Stack embeddings into matrix [num_nodes, 768]
            x = torch.stack([embeddings_dict[node_id] for node_id in node_ids])
            
            # Get edge_index [2, num_edges]
            edge_index = data_dict.get('edge_index', torch.empty((2, 0), dtype=torch.long))
            
            # Get label
            y = data_dict['label']
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                # Store metadata for debugging
                function_name=data_dict.get('function_name', ''),
                graph_type=data_dict.get('graph_type', ''),
                num_nodes=len(node_ids),
                num_edges=edge_index.shape[1] if edge_index.numel() > 0 else 0
            )
            
            return data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return empty graph on error
            return Data(
                x=torch.zeros((1, 768)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                y=torch.tensor(0, dtype=torch.long)
            )


class StratifiedEmbeddingDataset(EmbeddingDataset):
    """
    Dataset with balanced sampling of vulnerable/non-vulnerable functions.
    Useful when you have class imbalance (~2% vulnerable).
    """
    
    def __init__(self, *args, balance_ratio=1.0, **kwargs):
        """
        Args:
            balance_ratio: Ratio of vulnerable to non-vulnerable samples to keep.
                          1.0 = equal numbers, 0.5 = half as many vulnerable, etc.
        """
        super().__init__(*args, **kwargs)
        self.balance_ratio = balance_ratio
        self._rebalance()
    
    def _rebalance(self):
        """Filter file paths to balance classes."""
        print("Analyzing class distribution...")
        vulnerable = []
        non_vulnerable = []
        
        for fp in tqdm(self.file_paths, desc="Scanning labels"):
            try:
                data_dict = torch.load(fp, map_location='cpu')
                label = data_dict['label'].item()
                if label == 1:
                    vulnerable.append(fp)
                else:
                    non_vulnerable.append(fp)
            except:
                continue
        
        print(f"Original: {len(vulnerable)} vulnerable, {len(non_vulnerable)} non-vulnerable")
        
        # Balance by downsampling majority class
        target_non_vulnerable = int(len(vulnerable) / self.balance_ratio)
        
        if target_non_vulnerable < len(non_vulnerable):
            import random
            random.seed(42)
            non_vulnerable = random.sample(non_vulnerable, target_non_vulnerable)
        
        self.file_paths = vulnerable + non_vulnerable
        import random
        random.shuffle(self.file_paths)
        
        print(f"Balanced: {len(vulnerable)} vulnerable, {len(non_vulnerable)} non-vulnerable")
        print(f"Total samples: {len(self.file_paths)}")


# Example usage
if __name__ == "__main__":
    # Test loading
    print("Testing EmbeddingDataset...")
    
    # Load CFG graphs only
    dataset = EmbeddingDataset(
        root_dir='dataset_final_embeddings',
        graph_types=['cfg']
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Load first sample
    data = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Label: {data.y.item()}")
    print(f"  Function: {data.function_name}")
    
    # Test with train/val/test split
    from torch_geometric.loader import DataLoader
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_size}")
    print(f"  Val: {val_size}")
    print(f"  Test: {test_size}")
    
    # Test batch loading
    batch = next(iter(train_loader))
    print(f"\nBatch shape:")
    print(f"  x: {batch.x.shape}")
    print(f"  edge_index: {batch.edge_index.shape}")
    print(f"  y: {batch.y.shape}")
    print(f"  batch: {batch.batch.shape}")
