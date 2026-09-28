import pandas as pd
import torch
import os
import re
import argparse
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import json

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Create embeddings with graph structure')
parser.add_argument('--update-edges', action='store_true', 
                    help='Update existing embeddings with edge information without regenerating')
parser.add_argument('--csv', type=str, default=None, 
                    help='Path to a batch CSV file to process (overrides default)')
args = parser.parse_args()

# Configuration
BASE_PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT_FOLDER = "dataset_final_embeddings"
MODEL_NAME = "microsoft/graphcodebert-base"
MANIFEST_PATH = "dataset_manifest.csv"

GRAPH_TYPES = ["pdg"] #change depending on graph type (ast,cfg,pdg)

# Pooling strategy: 'cls' (default), 'mean', or 'max'
POOLING_STRATEGY = 'mean'  # Changed from 'cls' for better sequence representation

# Batch processing configuration
BATCH_SIZE = 64  # Process multiple nodes at once for speed (increased for CPU efficiency)

print("Loading GraphCodeBERT...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if args.update_edges:
    print("\n UPDATE MODE: Will add edge structure to existing embeddings")
    print("   (No embeddings will be regenerated)\n")
else:
    # CPU optimizations
    torch.set_num_threads(os.cpu_count())  # Use all CPU cores
    print(f"Using {os.cpu_count()} CPU threads")
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

# Initialize manifest file if it doesn't exist or in normal mode
if not args.update_edges:
    if not os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'w') as f:
            f.write('embedding_path,graph_type,label,function_name,commit,filepath\n')
        print(f"Created new manifest: {MANIFEST_PATH}")
    else:
        print(f"Appending to existing manifest: {MANIFEST_PATH}")

def append_to_manifest(embedding_path, graph_type, label, function_name, commit, filepath):
    """Append a single row to the manifest CSV."""
    with open(MANIFEST_PATH, 'a') as f:
        # Escape commas in fields
        function_name = function_name.replace(',', ';')
        filepath = filepath.replace(',', ';')
        f.write(f'{embedding_path},{graph_type},{label},{function_name},{commit},{filepath}\n')

def generate_embeddings_batch(code_list):
    """Generate embeddings for multiple code snippets at once (batch processing)."""
    if not code_list:
        return []
    
    # Filter out empty codes and track indices
    valid_codes = []
    valid_indices = []
    for i, code in enumerate(code_list):
        if code and len(code.strip()) > 0:
            valid_codes.append(code)
            valid_indices.append(i)
    
    if not valid_codes:
        return [torch.zeros(768) for _ in code_list]
    
    # Batch tokenization
    inputs = tokenizer(
        valid_codes, # list of code strings to tokenize
        return_tensors="pt", # output format
        truncation=True, # sequences > 512 tokens get cut off (this is the BERT architecture limit)
        max_length=512,
        padding=True # makes all sequences in batch the same length, because batch needs uniform shapes for matrix operations
    ).to(device) #GPU, if available
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Apply pooling strategy
    if POOLING_STRATEGY == 'cls':
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()
    elif POOLING_STRATEGY == 'mean':
        # Mean pooling with attention mask
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        embeddings = embeddings.cpu()
    elif POOLING_STRATEGY == 'max':
        embeddings = outputs.last_hidden_state.max(dim=1)[0].cpu()
    else:
        embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # Default to cls
    
    # Map embeddings back to original indices
    result = [torch.zeros(768) for _ in code_list]
    for i, valid_idx in enumerate(valid_indices):
        result[valid_idx] = embeddings[i]
    
    return result

def parse_dot_edges(dot_path):
    """Extract edges from .dot file using regex."""
    if not os.path.exists(dot_path):
        return torch.empty((2, 0), dtype=torch.long), {}
    
    try:
        with open(dot_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract edges: "123" -> "456"
        edge_pattern = r'"(\d+)"\s*->\s*"(\d+)"'
        edges = re.findall(edge_pattern, content)
        
        if not edges:
            return torch.empty((2, 0), dtype=torch.long), {}
        
        # Build node_to_idx mapping
        unique_nodes = sorted(set([src for src, _ in edges] + [dst for _, dst in edges]))
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Convert to tensor format [2, num_edges]
        edge_index = torch.tensor(
            [[node_to_idx[src], node_to_idx[dst]] for src, dst in edges],
            dtype=torch.long
        ).t()
        
        return edge_index, node_to_idx
    
    except Exception as e:
        print(f"Warning: Failed to parse edges from {dot_path}: {e}")
        return torch.empty((2, 0), dtype=torch.long), {}

# --- MAIN LOOP FOR EACH GRAPH TYPE ---
for graph_type in GRAPH_TYPES:
    print(f"\n --- STARTING TYPE: {graph_type.upper()} ---")
    
    # 1. Define dynamic paths
    #csv_name = f"functions-{graph_type}-linux-filtered.csv"
    #csv_path = os.path.join("output-data", csv_name)
    
    if args.csv is not None:
       csv_path = args.csv
       csv_name = os.path.basename(csv_path)
       print(f"[BATCH MODE] Using batch CSV: {csv_path}")
    else:
       csv_name = f"functions-{graph_type}-linux-filtered.csv"
       csv_path = os.path.join("output-data", csv_name)
    # Create specific subfolder (e.g., dataset_final_embeddings/ast)
    output_type_folder = os.path.join(OUTPUT_ROOT_FOLDER, graph_type)
    os.makedirs(output_type_folder, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"WARNING: CSV not found {csv_path}. Skipping this type.")
        continue

    print(f"Reading CSV: {csv_name}")
    try:
        df = pd.read_csv(csv_path, sep=';')
        # Filter out rows where CFG_filepath is NaN or not a string
        df = df[df['CFG_filepath'].notna()]
        df = df[df['CFG_filepath'].apply(lambda x: isinstance(x, str))]
        print(f"Valid rows after filtering: {len(df)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        continue

    if len(df) == 0:
        print(f" No valid rows found in CSV")
        continue

    print(f"Processing {len(df)} functions...")

    # Error tracking
    success = 0
    errors = 0
    skipped = 0
    updated = 0
    missing_txt_files = []
    empty_graphs = []
    empty_code_nodes = 0
    label_counts = {0: 0, 1: 0}

    # Progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {graph_type}"):
        try:
            # A. Paths
            relative_dot_path = row['CFG_filepath']
            full_dot_path = os.path.join(BASE_PROJECT_FOLDER, relative_dot_path)
            txt_path = full_dot_path + ".txt"  # Get txt files for embeddings
            
            # Check if already processed
            safe_name = relative_dot_path.replace('/', '__').replace('.dot', '.pt')
            final_path = os.path.join(output_type_folder, safe_name)
            
            # UPDATE MODE: Add edges to existing file
            if args.update_edges:
                if os.path.exists(final_path):
                    try:
                        # Load existing embedding
                        existing_data = torch.load(final_path)
                        
                        # Check if already has edges
                        if 'edge_index' in existing_data:
                            skipped += 1
                            continue
                        
                        # Extract graph structure
                        edge_index, node_to_idx = parse_dot_edges(full_dot_path)
                        
                        # Add edge information to existing data
                        existing_data['edge_index'] = edge_index
                        existing_data['node_to_idx'] = node_to_idx
                        existing_data['num_edges'] = edge_index.shape[1]
                        
                        # Save updated data
                        torch.save(existing_data, final_path)
                        updated += 1
                        continue
                    except Exception as e:
                        print(f"\n Error updating {safe_name}: {e}")
                        errors += 1
                        continue
                else:
                    # File doesn't exist yet - skip in update mode
                    skipped += 1
                    continue
            
            # NORMAL MODE: Skip if already exists
            if os.path.exists(final_path):
                skipped += 1
                continue
            
            if not os.path.exists(txt_path):
                # Try alternative path
                txt_path_alt = full_dot_path.replace(".dot", ".txt")
                if os.path.exists(txt_path_alt):
                    txt_path = txt_path_alt
                else:
                    missing_txt_files.append(relative_dot_path)
                    errors += 1
                    continue

            # B. Read node codes and prepare for batch processing
            node_data = []  # List of (node_id, code)
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        node_id = parts[0]
                        code = parts[1]
                        node_data.append((node_id, code))
            
            if not node_data:
                empty_graphs.append(relative_dot_path)
                errors += 1
                continue
            
            # C. Generate embeddings in batches
            node_embeddings = {}
            for i in range(0, len(node_data), BATCH_SIZE):
                batch = node_data[i:i + BATCH_SIZE]
                batch_codes = [code for _, code in batch]
                batch_embeddings = generate_embeddings_batch(batch_codes)
                
                for (node_id, _), embedding in zip(batch, batch_embeddings):
                    node_embeddings[node_id] = embedding
                    if embedding.sum() == 0:  # Empty embedding
                        empty_code_nodes += 1

            # D. Extract graph structure (edges)
            edge_index, node_to_idx = parse_dot_edges(full_dot_path)
            
            # E. Prepare data
            label_int = 1 if str(row['vulnerable_label']).lower() == "true" else 0
            label_counts[label_int] += 1
            
            data_to_save = {
                'embeddings': node_embeddings,
                'label': torch.tensor(label_int, dtype=torch.long),
                'function_name': row['function_name'],
                'dot_path': relative_dot_path,
                'graph_type': graph_type,
                'commit': row['commit'],
                'filepath': row['filepath'],
                'num_nodes': len(node_embeddings),
                'pooling_strategy': POOLING_STRATEGY,
                # Graph structure for GNN
                'edge_index': edge_index,
                'node_to_idx': node_to_idx,
                'num_edges': edge_index.shape[1]
            }

            # F. Save in correct subfolder
            torch.save(data_to_save, final_path)
            success += 1
            
            # G. Append to manifest (only in normal mode, not update mode)
            if not args.update_edges:
                relative_embedding_path = os.path.join(graph_type, safe_name)
                append_to_manifest(
                    embedding_path=relative_embedding_path,
                    graph_type=graph_type,
                    label=label_int,
                    function_name=row['function_name'],
                    commit=row['commit'],
                    filepath=row['filepath']
                )

        except Exception as e:
            print(f" Unexpected error at row {index}: {e}")
            errors += 1

    # Statistics
    print(f"\n✅ {graph_type.upper()} Complete!")
    if args.update_edges:
        print(f"   Updated: {updated} files with edge structure")
        print(f"   Skipped: {skipped} (already have edges)")
    else:
        print(f"   Success: {success} files generated")
        print(f"   Skipped: {skipped} (already processed)")
    print(f"   Errors: {errors} ({errors/len(df)*100:.1f}%)")
    print(f"   Missing .txt files: {len(missing_txt_files)}")
    print(f"   Empty graphs: {len(empty_graphs)}")
    print(f"   Empty code nodes: {empty_code_nodes}")
    print(f"   Labels - Vulnerable: {label_counts[1]}, Non-vulnerable: {label_counts[0]}")
    if label_counts[0] + label_counts[1] > 0:
        print(f"   Class balance: {label_counts[1]/(label_counts[0]+label_counts[1])*100:.2f}% vulnerable")
    
    # Save error log
    if missing_txt_files or empty_graphs:
        error_log_path = os.path.join(output_type_folder, f"error_log_{graph_type}.json")
        with open(error_log_path, 'w') as f:
            json.dump({
                'missing_txt_files': missing_txt_files[:100],  # First 100 examples
                'empty_graphs': empty_graphs[:100],
                'total_missing': len(missing_txt_files),
                'total_empty': len(empty_graphs)
            }, f, indent=2)
        print(f"   Error log saved: {error_log_path}")

print(f"\nTotal embeddings created: {sum([len([f for f in os.listdir(os.path.join(OUTPUT_ROOT_FOLDER, gt)) if f.endswith('.pt')]) for gt in GRAPH_TYPES if os.path.exists(os.path.join(OUTPUT_ROOT_FOLDER, gt))])}")
