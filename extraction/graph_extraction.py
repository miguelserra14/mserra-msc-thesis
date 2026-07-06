import os
import subprocess
import html
import pydot

# Path configuration
JOERN_CLI_DIR = os.path.expanduser("~/joern-cli")
JOERN_PARSE = os.path.join(JOERN_CLI_DIR, "joern-parse")
JOERN_EXPORT = os.path.join(JOERN_CLI_DIR, "joern-export")

# Temporary file (Code Property Graph) that creates a bridge between joern-parse and joern-export
TEMP_CPG_PATH = "/tmp/temp_cpg.bin"

def obtain_directories(repository_path, run_per_directory):
    if run_per_directory:
        return [directory
                for directory in os.listdir(repository_path)
                if os.path.isdir(os.path.join(repository_path, directory)) and
                not directory.startswith(".")]
    return [""]

def generate_txt_force(dot_path):
    """
    .txt file immediately next to corresponding .dot file 
    """
    # files will follow the structure "0-cfg.dot.txt" and are next to corresponding dot .dot
    txt_path = dot_path + ".txt"
    
    try:
        graphs = pydot.graph_from_dot_file(dot_path)
        if not graphs: return
        graph = graphs[0]
        
        with open(txt_path, 'w') as f:
            for node in graph.get_nodes():
                node_id = node.get_name().strip('"')
                raw_label = node.get("label") # gets the label attribute from the .dot file, with metadata and HTML formatting.
                
                if raw_label:
                    # Cleaning part
                    clean_label = raw_label.strip('"').strip('<>')
                    if "<BR/>" in clean_label:
                        clean_label = clean_label.split("<BR/>")[-1]
                    elif "<BR>" in clean_label:
                        clean_label = clean_label.split("<BR>")[-1]
                    
                    final_code = html.unescape(clean_label).strip()
                    
                    # node id + clean code
                    f.write(f"{node_id} {final_code}\n")
        
        print(f"   [TXT] Created: {os.path.basename(txt_path)}") 

    except Exception as e:
        print(f"   [TXT ERROR] Failed in {dot_path}: {e}")
        
# Function in main.py
def extract_graph_per_file(base_output_directory, project, repository_path, commit, files, graph_type):
    
    # Add Joern to path
    os.environ["PATH"] += os.pathsep + JOERN_CLI_DIR

    output_directory = os.path.join(base_output_directory, project,
                                    "output-{}-{}".format(graph_type, commit))

    os.makedirs(output_directory, exist_ok=True)
        
    for filename in files:
        filepath = os.path.join(repository_path, filename)
        filepath_directory = filename.replace("/", "---")
        graph_output_directory = os.path.join(output_directory, filepath_directory)
        
        is_valid = False
        if os.path.exists(graph_output_directory):
            txt_files = [f for f in os.listdir(graph_output_directory) if f.endswith(".txt")]
            for txt in txt_files:
                txt_path = os.path.join(graph_output_directory, txt)
                # Verifica se o ficheiro tem dados lá dentro (> 0 bytes)
                if os.path.getsize(txt_path) > 0:
                    is_valid = True
                    break
                    
        if is_valid:
            print(f"   [SKIP] Válido: {filename}")
            continue 

        print(f"   > Processing: {filename}")

        # Prse
        cmd_parse = [JOERN_PARSE, filepath, "--output", TEMP_CPG_PATH]
        subprocess.run(cmd_parse, capture_output=True)
        
        # Export dot
        cmd_export = [JOERN_EXPORT, TEMP_CPG_PATH, "--repr", graph_type, "--out", graph_output_directory]
        subprocess.run(cmd_export, capture_output=True)

        # -Export txt
        if os.path.exists(graph_output_directory):
            for f in os.listdir(graph_output_directory):
                if f.endswith(".dot"):
                    full_dot_path = os.path.join(graph_output_directory, f)
                    generate_txt_force(full_dot_path)

        # Clean temporary CPGs
        if os.path.exists(TEMP_CPG_PATH):
            os.remove(TEMP_CPG_PATH)

    return output_directory
