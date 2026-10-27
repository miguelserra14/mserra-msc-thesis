# old luis
import os
import pandas as pd
import networkx as nx
import html
from networkx.drawing.nx_pydot import from_pydot


from graph_extraction_constants import GRAPH_FILE
from graph_parsing import read_graph
from reduce_graph import reduce_graph, write_dot_file, write_statement_file

projects = ["httpd", "glibc", "gecko-dev", "linux", "xen"]
data_directory = "output-data"
file_cfg_data_mask = "functions-{}-{}.csv"
#graph = "cfg"

def check_created_directory(base_directory, type_directory, output_commit, repository_directory):
    output_directory = os.path.join(base_directory, type_directory, output_commit, repository_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory

import html # Garante que tens isto

def generate_unreduced_graph_artifacts(project, graph_type):
    print(f"\n--- 🕵️‍♂️ DEBUG: A iniciar artefactos para {project} ({graph_type}) ---")
    
    # 1. CONSTRUÇÃO DO NOME DO FICHEIRO (A Forma Segura)
    # Em vez de máscaras confusas, vamos construir o nome explicitamente
    # O teu ficheiro chama-se: functions-cfg-linux.csv
    csv_filename = f"functions-{graph_type}-{project}.csv"
    filepath = os.path.join(data_directory, csv_filename)
    
    print(f"📍 À procura do ficheiro: {filepath}")

    # 2. VERIFICAR SE EXISTE
    if not os.path.exists(filepath):
        print(f"❌ ERRO CRÍTICO: O ficheiro não existe neste caminho!")
        print(f"   (Verifica se a pasta '{data_directory}' está no sítio certo)")
        return

    # 3. TENTAR LER
    try:
        # Nota: O código original usava delimiter=";" - Verifica se o teu CSV usa ; ou ,
        # Vamos tentar ler a primeira linha para ver o separador
        with open(filepath, 'r') as f:
            header = f.readline()
        print(f"📄 Cabeçalho do ficheiro: {header.strip()}")
        
        separator = ";" if ";" in header else ","
        print(f"⚙️ Detetado separador: '{separator}'")

        df = pd.read_csv(filepath, delimiter=separator)
        print(f"📊 Total linhas lidas: {len(df)}")
        
        # Filtrar nulos
        df = df[df[GRAPH_FILE].notnull()]
        print(f"✅ Linhas válidas (com grafo): {len(df)}")
        
        if len(df) == 0:
            print("⚠️ AVISO: O ficheiro existe mas não tem linhas válidas! O passo anterior (map_functions) falhou?")
            return

    except Exception as e:
        print(f"❌ ERRO AO LER CSV: {e}")
        return

    # 4. PROCESSAMENTO (O Loop Principal)
    count = 0
    print("🚀 A iniciar processamento dos nós...")
    
    for index, row in df.iterrows():
        try:
            graph_filepath = row[GRAPH_FILE]
            
            # --- (Lógica de caminhos original) ---
            graph_filepath_parts = graph_filepath.split("\\" if "\\" in graph_filepath else "/")
            graph_filename = graph_filepath_parts[-1]
            graph_name = graph_filename.replace(".dot", "")

            if project != "httpd":
                repository_directory = graph_filepath_parts[-2]
                output_commit_index = -3
                base_directory_max_index = -4
            else:
                repository_directory = ""
                output_commit_index = -2
                base_directory_max_index = -3

            output_commit = graph_filepath_parts[output_commit_index]
            base_directory = "/".join(graph_filepath_parts[:base_directory_max_index])
            # -------------------------------------

            graphs = read_graph(graph_filepath, print_filepath=False) # print_filepath=False para menos lixo
            
            if graphs is not None:
                graph_dot = graphs[0]
                graph_nx = nx.nx_pydot.from_pydot(graph_dot)

                # Criar diretorias
                output_statements_directory = check_created_directory(
                    base_directory, f"{project}-{graph_type}-statements", output_commit, repository_directory)

                # Extrair e Limpar
                node_statements = {}
                for node in graph_nx.nodes():
                    try:
                        dot_node = graph_dot.get_node(f'"{node}"')
                        if dot_node and dot_node[0].get_label():
                            raw_label = dot_node[0].get_label()
                            
                            # LIMPEZA
                            clean_label = raw_label.strip('"').strip('<>')
                            if "<BR/>" in clean_label:
                                parts = clean_label.split("<BR/>")
                                clean_label = parts[-1] 
                            elif "<BR>" in clean_label:
                                parts = clean_label.split("<BR>")
                                clean_label = parts[-1]
                            
                            final_code = html.unescape(clean_label).strip()
                            node_statements[str(node)] = [final_code]
                        else:
                            node_statements[str(node)] = []
                    except:
                        node_statements[str(node)] = []

                write_statement_file(output_statements_directory, graph_name, node_statements)
                count += 1
                
                # Feedback visual a cada 100 ficheiros
                if count % 100 == 0:
                    print(f"   ...processados {count} grafos.")

        except Exception as e:
            print(f"⚠️ Erro ao processar linha {index}: {e}")
            continue

    print(f"🏁 CONCLUÍDO: Gerados {count} ficheiros .txt com sucesso!")


'''
def generate_unreduced_graph_artifacts(project, graph_type):
    # filepath = os.path.join(data_directory, file_cfg_data_mask.format(project,graph_type))
    filepath = os.path.join(data_directory, f"functions-{graph_type}-{project}.csv")
    df = pd.read_csv(filepath, delimiter=";")
    df = df[df[CFG_FILE].notnull()]

    for index, row in df.iterrows():
        graph_filepath = row[CFG_FILE]

        # Caminho dividido em partes
        graph_filepath_parts = graph_filepath.split("\\" if "\\" in graph_filepath else "/")
        graph_filename = graph_filepath_parts[-1]
        graph_name = graph_filename.replace(".dot", "")

        if project != "httpd":
            repository_directory = graph_filepath_parts[-2]
            output_commit_index = -3
            base_directory_max_index = -4
        else:
            repository_directory = ""
            output_commit_index = -2
            base_directory_max_index = -3

        output_commit = graph_filepath_parts[output_commit_index]
        base_directory = "/".join(graph_filepath_parts[:base_directory_max_index])

        graphs = read_graph(graph_filepath)
        if graphs is not None:
            graph_dot = graphs[0]
            graph_nx = nx.nx_pydot.from_pydot(graph_dot)

            output_reduced_directory = check_created_directory(
                base_directory, f"{project}-{graph_type}-reduced", output_commit, repository_directory)
            output_statements_directory = check_created_directory(
                base_directory, f"{project}-{graph_type}-statements", output_commit, repository_directory)

            write_dot_file(output_reduced_directory, graph_name, graph_nx)

            # Extrai labels para cada nó
            node_statements = {}
            for node in graph_nx.nodes():
                try:
                    dot_node = graph_dot.get_node(f'"{node}"')
                    if dot_node and dot_node[0].get_label():
                        raw_label = dot_node[0].get_label()
                        
                        # --- INICIO DA LIMPEZA ---
                        # 1. Remove aspas e as tags HTML de fora (<...>)
                        clean_label = raw_label.strip('"').strip('<>')
                        
                        # 2. O Joern separa o Tipo do Código com <BR/>. Nós queremos a parte direita (o código).
                        # Exemplo: "OPERATOR, 45<BR/>x = y"  -> queremos "x = y"
                        if "<BR/>" in clean_label:
                            parts = clean_label.split("<BR/>")
                            clean_label = parts[-1] 
                        elif "<BR>" in clean_label: # Caso raro, mas acontece
                            parts = clean_label.split("<BR>")
                            clean_label = parts[-1]

                        # 3. Converte caracteres HTML (&lt; vira <, &gt; vira >)
                        final_code = html.unescape(clean_label).strip()
                        # --- FIM DA LIMPEZA ---

                        node_statements[str(node)] = [final_code]
                    else:
                        node_statements[str(node)] = []
                except Exception as e:
                    print(f"Warning: failed to get label for node {node}: {e}")
                    node_statements[str(node)] = []



'''
def reduce_read_cfg_file(project,graph_type="cfg"):
    filepath = os.path.join(data_directory, file_cfg_data_mask.format(project,graph_type))
    df = pd.read_csv(filepath, delimiter=";")
    df = df[df[CFG_FILE].notnull()]

    for index, row in df.iterrows():
        cfg_filepath = row[CFG_FILE]

        cfg_filepath_parts = cfg_filepath.split("\\" if "\\" in cfg_filepath else "/")
        cfg_filename = cfg_filepath_parts[-1]
        cfg_name = cfg_filename[:cfg_filename.index(".dot")]

        if project != "httpd":
            repository_directory = cfg_filepath_parts[-2]
            output_commit_index = -3
            base_directory_max_index = -4
        else:
            repository_directory = ""
            output_commit_index = -2
            base_directory_max_index = -3
        output_commit = cfg_filepath_parts[output_commit_index]
        base_directory = "/".join(cfg_filepath_parts[0:base_directory_max_index])

        graphs = read_graph(cfg_filepath)

        if graphs is not None:
            cfg_dot = graphs[0]
            cfg_nx_merged, node_statements = reduce_graph(cfg_dot)

            output_reduced_directory = check_created_directory(
                base_directory, "{}-{}-reduced".format(project,graph_type), output_commit, repository_directory)
            output_statements_directory = check_created_directory(
                base_directory, "{}-{}-statements".format(project,graph_type), output_commit, repository_directory)

            write_dot_file(output_reduced_directory, cfg_name, cfg_nx_merged)
            write_statement_file(output_statements_directory, cfg_name, node_statements)


def check_created_directory(base_directory, type_directory, output_commit, repository_directory):
    output_directory = os.path.join(base_directory, type_directory, output_commit, repository_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory


def main():
    for project in projects[3:4]:
        reduce_read_cfg_file(project)


if __name__ == "__main__":
    main()

