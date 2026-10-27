"""
Validation script for embedding dataset.
Checks data integrity, statistics, and provides insights.
"""
import torch
import os
import json
from collections import defaultdict
import numpy as np

OUTPUT_ROOT_FOLDER = "dataset_final_embeddings"
GRAPH_TYPES = ["ast", "cfg", "pdg"]

def validate_embeddings():
    print("Embedding dataset Validation")
    
    all_stats = {}
    
    for graph_type in GRAPH_TYPES:
        print(f"\n{'='*60}")
        print(f"Analyzing {graph_type.upper()}")
        print(f"{'='*60}")
        
        type_folder = os.path.join(OUTPUT_ROOT_FOLDER, graph_type)
        
        if not os.path.exists(type_folder):
            print(f"Folder doesn't exist: {type_folder}")
            continue
        
        files = [f for f in os.listdir(type_folder) if f.endswith('.pt')]
        
        if not files:
            print(f"No .pt files found")
            continue
        
        print(f"\nTotal files: {len(files)}")
        
        # Statistics collectors
        labels = []
        num_nodes_list = []
        embedding_dims = []
        function_names = []
        commits = set()
        corrupted_files = []
        
        print("   Processing files")
        for i, filename in enumerate(files):
            if i % 100 == 0 and i > 0:
                print(f"   processed {i}/{len(files)}")
            
            filepath = os.path.join(type_folder, filename)
            try:
                data = torch.load(filepath, map_location='cpu')
                
                # Validate structure
                required_keys = ['embeddings', 'label', 'function_name', 'dot_path', 'graph_type']
                for key in required_keys:
                    if key not in data:
                        corrupted_files.append((filename, f"Missing key: {key}"))
                        continue
                
                # Collect statistics
                labels.append(data['label'].item())
                num_nodes_list.append(len(data['embeddings']))
                function_names.append(data['function_name'])
                
                if 'commit' in data:
                    commits.add(data['commit'])
                
                # Check embedding dimensions
                if data['embeddings']:
                    first_emb = list(data['embeddings'].values())[0]
                    embedding_dims.append(first_emb.shape[0])
                
            except Exception as e:
                corrupted_files.append((filename, str(e)))
        
        # Calculate statistics
        labels_array = np.array(labels)
        num_nodes_array = np.array(num_nodes_list)
        
        print("Statistics")
        
        print(f"\nLabels (Vulnerability):")
        print(f"   Total functions: {len(labels)}")
        print(f"   Vulnerable (1): {np.sum(labels_array)} ({np.mean(labels_array)*100:.2f}%)")
        print(f"   Non-vulnerable (0): {len(labels_array) - np.sum(labels_array)} ({(1-np.mean(labels_array))*100:.2f}%)")
        
        print(f"\nNodes per Graph:")
        print(f"   Mean: {np.mean(num_nodes_array):.1f} nodes")
        print(f"   Median: {np.median(num_nodes_array):.0f} nodes")
        print(f"   Min-Max: {np.min(num_nodes_array)}-{np.max(num_nodes_array)} nodes")
        print(f"   Std: {np.std(num_nodes_array):.1f}")
        
        
        print(f"\Embeddings:")
        if embedding_dims:
            unique_dims = set(embedding_dims)
            print(f"   Dimensions: {unique_dims}")
        
        print(f"\nUnique commits: {len(commits)}")
        print(f"Unique functions: {len(set(function_names))}")
        
        if corrupted_files:
            print(f"\nCorrupted files: {len(corrupted_files)}")
            print("   First examples:")
            for fname, error in corrupted_files[:5]:
                print(f"      - {fname}: {error}")
        
        # Save statistics
        stats = {
            'graph_type': graph_type,
            'total_files': len(files),
            'total_functions': len(labels),
            'vulnerable': int(np.sum(labels_array)),
            'non_vulnerable': int(len(labels_array) - np.sum(labels_array)),
            'vulnerable_ratio': float(np.mean(labels_array)),
            'nodes_mean': float(np.mean(num_nodes_array)),
            'nodes_median': float(np.median(num_nodes_array)),
            'nodes_min': int(np.min(num_nodes_array)),
            'nodes_max': int(np.max(num_nodes_array)),
            'nodes_std': float(np.std(num_nodes_array)),
            'unique_commits': len(commits),
            'unique_functions': len(set(function_names)),
            'corrupted_files': len(corrupted_files),
            'embedding_dims': list(set(embedding_dims)) if embedding_dims else []
        }
        
        all_stats[graph_type] = stats
    
    # Save overall statistics
    stats_file = os.path.join(OUTPUT_ROOT_FOLDER, "dataset_statistics.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"Statistics saved in: {stats_file}")

    # Overall summary
    print(f"\nOverall Summary")
    total_embeddings = sum(s['total_files'] for s in all_stats.values())
    total_vulnerable = sum(s['vulnerable'] for s in all_stats.values())
    total_functions = sum(s['total_functions'] for s in all_stats.values())
    
    print(f"Total embeddings: {total_embeddings}")
    print(f"Total functions: {total_functions}")
    print(f"Total vulnerable: {total_vulnerable} ({total_vulnerable/total_functions*100:.2f}%)")
    
    print(f"\nPor tipo de grafo:")
    for gt in GRAPH_TYPES:
        if gt in all_stats:
            s = all_stats[gt]
            print(f"  {gt.upper()}: {s['total_files']} embeddings, "
                  f"{s['vulnerable']}/{s['total_functions']} vuln ({s['vulnerable_ratio']*100:.1f}%)")


def check_sample_embeddings():
    """Load and display a few sample embeddings for inspection."""
    print(f"\n{'='*60}")
    print("🔬 INSPEÇÃO DE AMOSTRAS")
    print(f"{'='*60}")
    
    for graph_type in GRAPH_TYPES:
        pasta_tipo = os.path.join(PASTA_SAIDA_RAIZ, graph_type) # type: ignore
        if not os.path.exists(pasta_tipo):
            continue
        
        files = [f for f in os.listdir(pasta_tipo) if f.endswith('.pt')][:3]
        
        if not files:
            continue
        
        print(f"\n{graph_type.upper()} - Amostra:")
        for fname in files:
            try:
                data = torch.load(os.path.join(pasta_tipo, fname), map_location='cpu')
                print(f"\n {fname}")
                print(f"     Function: {data['function_name']}")
                print(f"     Label: {'VULN' if data['label'].item() == 1 else 'NON-VULN'}")
                print(f"     Nodes: {len(data['embeddings'])}")
                if 'commit' in data:
                    print(f"     Commit: {data['commit'][:12]}")
                if 'pooling_strategy' in data:
                    print(f"     Pooling: {data['pooling_strategy']}")
            except Exception as e:
                print(f" Error reading {fname}: {e}")


if __name__ == "__main__":
    validate_embeddings()
    check_sample_embeddings()

