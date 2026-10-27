import os
import pandas as pd

DATA_FOLDER = "dataset_final_embeddings"
OUTPUT_CSV = "dataset_manifest.csv"


entries = []

for graph_type in ["ast", "cfg", "pdg"]:
    type_folder = os.path.join(DATA_FOLDER, graph_type)
    if not os.path.exists(type_folder): continue
    
    files = os.listdir(type_folder)
    for f in files:
        if f.endswith(".pt"):
            # Relative path that the DatasetLoader will use
            relative_path = os.path.join(graph_type, f)
            
            # Note: Label is already INSIDE the .pt file (data['label']),
            # but the manifest helps filter without loading the file.
            entries.append({
                'file_path': relative_path,
                'graph_type': graph_type
            })

df = pd.DataFrame(entries)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Manifest created with {len(df)} entries: {OUTPUT_CSV}")