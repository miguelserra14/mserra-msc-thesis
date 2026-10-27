"""
Update CSV mappings to include all newly extracted .dot files from Joern.
This regenerates functions-{graph_type}-linux.csv with current extraction status.
"""
from convert_dot_to_json import map_functions_to_graph

project = "linux"
graph_type_list = ["ast", "cfg", "pdg"]
csv_filename = "function-data/linux-functions.csv"

print("Updating CSV mappings for all newly extracted graphs...")
print(f"Input: {csv_filename}")
print(f"Output: output-data/functions-{{graph_type}}-{project}.csv\n")

for graph_type in graph_type_list:
    print(f"\n{'='*60}")
    print(f"Processing {graph_type.upper()}...")
    print(f"{'='*60}")
    
    map_functions_to_graph(project, graph_type, csv_filename=csv_filename)
    
    print(f"✅ {graph_type.upper()} CSV updated")

print("\n" + "="*60)
print("All CSV files updated successfully!")
print("="*60)
print("\nYou can now re-run embedding generation with the updated CSVs.")
