"""
Dataset preparation utilities for training.
Handles data loading, splitting, and balancing.
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from collections import Counter
import random

class GraphEmbeddingDataset(Dataset):
    """PyTorch Dataset for graph embeddings."""
    
    def __init__(self, embedding_dir, graph_type, transform=None):
        """
        Args:
            embedding_dir: Base directory (e.g., 'dataset_final_embeddings')
            graph_type: 'ast', 'cfg', or 'pdg'
            transform: Optional transform to apply to embeddings
        """
        self.embedding_dir = os.path.join(embedding_dir, graph_type)
        self.graph_type = graph_type
        self.transform = transform
        
        # Load all .pt files
        self.files = [f for f in os.listdir(self.embedding_dir) if f.endswith('.pt')]
        print(f"Loaded {len(self.files)} files for {graph_type}")
        
        # Cache metadata for quick access
        self.labels = []
        self.num_nodes = []
        
        print("Reading metadata...")
        for fname in self.files:
            try:
                data = torch.load(os.path.join(self.embedding_dir, fname), map_location='cpu')
                self.labels.append(data['label'].item())
                self.num_nodes.append(len(data['embeddings']))
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        
        # Statistics
        label_counts = Counter(self.labels)
        print(f"Label distribution: {dict(label_counts)}")
        print(f"Vulnerable ratio: {label_counts[1]/len(self.labels)*100:.2f}%")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """Load and return embedding data."""
        filepath = os.path.join(self.embedding_dir, self.files[idx])
        data = torch.load(filepath, map_location='cpu')
        
        # Convert node embeddings dict to tensor
        # Shape: [num_nodes, embedding_dim]
        node_ids = sorted(data['embeddings'].keys())
        embeddings = torch.stack([data['embeddings'][nid] for nid in node_ids])
        
        if self.transform:
            embeddings = self.transform(embeddings)
        
        return {
            'embeddings': embeddings,
            'label': data['label'],
            'num_nodes': len(node_ids),
            'function_name': data['function_name'],
            'graph_type': data['graph_type']
        }
    
    def get_class_weights(self):
        """Calculate class weights for weighted loss."""
        label_counts = Counter(self.labels)
        total = len(self.labels)
        
        # Weight inversely proportional to class frequency
        weights = torch.tensor([
            total / (2 * label_counts[0]),  # Non-vulnerable
            total / (2 * label_counts[1])   # Vulnerable
        ])
        
        return weights
    
    def get_sampler(self):
        """Create weighted sampler for balanced batching."""
        label_counts = Counter(self.labels)
        
        # Weight each sample inversely to its class frequency
        sample_weights = [1.0 / label_counts[label] for label in self.labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def collate_graph_batch(batch):
    """
    Custom collate function for variable-sized graphs.
    Pads embeddings to the same size within a batch.
    """
    max_nodes = max(item['num_nodes'] for item in batch)
    embedding_dim = batch[0]['embeddings'].shape[1]
    batch_size = len(batch)
    
    # Create padded tensor
    padded_embeddings = torch.zeros(batch_size, max_nodes, embedding_dim)
    masks = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)
    
    for i, item in enumerate(batch):
        num_nodes = item['num_nodes']
        padded_embeddings[i, :num_nodes] = item['embeddings']
        masks[i, :num_nodes] = True
        labels[i] = item['label']
    
    return {
        'embeddings': padded_embeddings,  # [batch_size, max_nodes, embedding_dim]
        'masks': masks,                    # [batch_size, max_nodes]
        'labels': labels,                  # [batch_size]
        'num_nodes': [item['num_nodes'] for item in batch]
    }


def create_dataloaders(embedding_dir='dataset_final_embeddings', 
                       graph_type='cfg',
                       batch_size=32,
                       train_split=0.8,
                       use_weighted_sampling=True,
                       seed=42):
    """
    Create train and validation dataloaders.
    
    Args:
        embedding_dir: Base directory with embeddings
        graph_type: Type of graph ('ast', 'cfg', 'pdg')
        batch_size: Batch size
        train_split: Proportion of data for training
        use_weighted_sampling: Whether to use weighted sampling for balance
        seed: Random seed
    
    Returns:
        train_loader, val_loader, class_weights
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset
    dataset = GraphEmbeddingDataset(embedding_dir, graph_type)
    
    # Split indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create dataloaders
    if use_weighted_sampling:
        # Get sampler only for training set
        train_labels = [dataset.labels[i] for i in train_indices]
        label_counts = Counter(train_labels)
        sample_weights = [1.0 / label_counts[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=collate_graph_batch,
            num_workers=2
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_graph_batch,
            num_workers=2
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_graph_batch,
        num_workers=2
    )
    
    # Get class weights for loss function
    class_weights = dataset.get_class_weights()
    
    print(f"\nClass weights for loss: {class_weights}")
    
    return train_loader, val_loader, class_weights


def test_dataloader():
    """Test function to verify dataloader works."""
    print("="*60)
    print("🧪 TESTING DATALOADER")
    print("="*60)
    
    for graph_type in ['ast', 'cfg', 'pdg']:
        print(f"\n--- Testing {graph_type.upper()} ---")
        
        embedding_dir = 'dataset_final_embeddings'
        if not os.path.exists(os.path.join(embedding_dir, graph_type)):
            print(f"Skipping {graph_type} - directory not found")
            continue
        
        try:
            train_loader, val_loader, class_weights = create_dataloaders(
                embedding_dir=embedding_dir,
                graph_type=graph_type,
                batch_size=4,
                train_split=0.8
            )
            
            # Test one batch
            print("\nTesting one batch from train_loader")
            batch = next(iter(train_loader))
            
            print(f"Batch embeddings shape: {batch['embeddings'].shape}")
            print(f"Batch masks shape: {batch['masks'].shape}")
            print(f"Batch labels shape: {batch['labels'].shape}")
            print(f"Labels in batch: {batch['labels'].tolist()}")
            print(f"Num nodes per graph: {batch['num_nodes']}")
            
            print(f"\n{graph_type.upper()} dataloader works")
            
        except Exception as e:
            print(f"Error with {graph_type}: {e}")


if __name__ == "__main__":
    test_dataloader()
