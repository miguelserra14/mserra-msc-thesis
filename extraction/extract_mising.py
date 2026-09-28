import os
import subprocess
import html
import pydot

# Path configuration
JOERN_CLI_DIR = os.path.expanduser("~/joern-cli")
JOERN_PARSE = os.path.join(JOERN_CLI_DIR, "joern-parse")
JOERN_EXPORT = os.path.join(JOERN_CLI_DIR, "joern-export")
TEMP_CPG_PATH = "/tmp/temp_cpg.bin"

def generate_txt_force(dot_path):
    txt_path = dot_path + ".txt"
    try:
        graphs = pydot.graph_from_dot_file(dot_path)
        if not graphs: return
        graph = graphs[0]
        
        with open(txt_path, 'w') as f:
            for node in graph.get_nodes():
                node_id = node.get_name().strip('"')
                raw_label = node.get("label")
                
                if raw_label:
                    clean_label = raw_label.strip('"').strip('<>')
                    if "<BR/>" in clean_label:
                        clean_label = clean_label.split("<BR/>")[-1]
                    elif "<BR>" in clean_label:
                        clean_label = clean_label.split("<BR>")[-1]
                    
                    final_code = html.unescape(clean_label).strip()
                    f.write(f"{node_id} {final_code}\n")
        
        print(f"   [TXT] Criado: {os.path.basename(txt_path)}") 

    except Exception as e:
        print(f"   [TXT ERROR] Falhou em {dot_path}: {e}")
        
def extract_graph_per_file(base_output_directory, project, repository_path, commit, files, graph_type):
    os.environ["PATH"] += os.pathsep + JOERN_CLI_DIR
    output_directory = os.path.join(base_output_directory, project, f"output-{graph_type}-{commit}")
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
                # Verifies if file exists and if there is data(> 0 bytes)
                if os.path.getsize(txt_path) > 0:
                    is_valid = True
                    break
        
        if is_valid:
            print(f"[SKIP] Already exists and is valid: {filename}")
            continue 
            
        # if it gets here, the file is missing or corrupted
        print(f"\n[EXTRACTING] Extracting missing/corrupted file: {filename}")

        cmd_parse = [JOERN_PARSE, filepath, "--output", TEMP_CPG_PATH]
        subprocess.run(cmd_parse, capture_output=True)
        
        cmd_export = [JOERN_EXPORT, TEMP_CPG_PATH, "--repr", graph_type, "--out", graph_output_directory]
        subprocess.run(cmd_export, capture_output=True)

        if os.path.exists(graph_output_directory):
            for f in os.listdir(graph_output_directory):
                if f.endswith(".dot"):
                    full_dot_path = os.path.join(graph_output_directory, f)
                    generate_txt_force(full_dot_path)

        if os.path.exists(TEMP_CPG_PATH):
            os.remove(TEMP_CPG_PATH)

    return output_directory

if __name__ == "__main__":
    BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    PROJECT = "linux"
    REPO_PATH = os.path.expanduser("~/tese_workspace/linux")
    COMMIT = "v0.5" 
    GRAPH_TYPE = "cfg"
    todos_os_ficheiros = [
        "drivers/usb/core/devio.c",
    ]

    extract_graph_per_file(BASE_OUTPUT_DIR, PROJECT, REPO_PATH, COMMIT, todos_os_ficheiros, GRAPH_TYPE)