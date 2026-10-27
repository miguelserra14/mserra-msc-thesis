"""Corre a pipeline ORIGINAL da Constança sobre dados/vuln.c e deixa tudo em _gerado/pipeline/.

Passos (os mesmos do main.py + create_embeddings.py, com o ficheiro de teste no lugar do Linux):
  1. Joern: parse + export AST/CFG/PDG              -> output/linux/output-{tipo}-{commit}/drivers---test---vuln.c/
  2. graph_extraction.generate_txt_force (original)  -> N-{tipo}.dot.txt
  3. convert_dot_to_json.map_single_commit_to_graph (original), corrido 2x como numa reexecução
                                                    -> output-data/functions-{tipo}-linux.csv
  4. passo manual da Constança (sem script no repo)  -> functions-{tipo}-linux-filtered.csv (coluna CFG_filepath)
  5. create_embeddings.py (original; só muda GRAPH_TYPES, como ela fazia à mão; modelo substituído por stub)
                                                    -> dataset_final_embeddings/{tipo}/*.pt
"""
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (COMMIT_TESTE, DADOS, EXTRACAO, FICHEIRO_TESTE, PIPELINE, TIPOS, cabecalho, correr_joern,
                   nomes_dos_dots, pasta_pipeline, secao)

# Funções de dados/vuln.c tal como o dataset de origem as listaria (Name, Lines, Vulnerable).
# As duas `helper` vêm dos dois ramos do #ifdef; a segunda é marcada vulnerável de propósito.
FUNCOES = [
    {"Name": "copy_item", "Signature": "copy_item(struct item *, const struct item *)", "Lines": [9, 20], "Vulnerable": "Yes"},
    {"Name": "helper", "Signature": "helper(int)", "Lines": [24, 24], "Vulnerable": "No"},
    {"Name": "helper", "Signature": "helper(int)", "Lines": [26, 30], "Vulnerable": "Yes"},
    {"Name": "log_msg", "Signature": "log_msg(const char *)", "Lines": [34, 37], "Vulnerable": "No"},
    {"Name": "compute", "Signature": "compute(int, int)", "Lines": [40, 47], "Vulnerable": "No"},
    {"Name": "get_len", "Signature": "get_len(struct item *)", "Lines": [51, 51], "Vulnerable": "No"},
    {"Name": "main", "Signature": "main(int, char **)", "Lines": [53, 63], "Vulnerable": "No"},
]


def main():
    cabecalho("00 - Preparação: pipeline original sobre dados/vuln.c")
    if PIPELINE.exists():
        shutil.rmtree(PIPELINE)
    PIPELINE.mkdir(parents=True)

    secao("1. Joern (parse + export ast/cfg/pdg)")
    saidas = correr_joern(DADOS / "vuln.c", TIPOS)
    for tipo in TIPOS:
        destino = pasta_pipeline(tipo)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(saidas[tipo], destino)
        nomes = nomes_dos_dots(destino)
        print(f"{tipo}: {len(nomes)} .dot -> " + ", ".join(f"{k.split('-')[0]}={v}" for k, v in list(nomes.items())[:9]) + ", ...")

    secao("2. generate_txt_force (original)")
    sys.path.insert(0, str(EXTRACAO))
    os.chdir(PIPELINE)
    from graph_extraction import generate_txt_force
    import contextlib, io
    n = 0
    for dot in sorted(PIPELINE.glob("output/linux/*/*/*.dot")):
        with contextlib.redirect_stdout(io.StringIO()):
            generate_txt_force(str(dot.relative_to(PIPELINE)))
        n += 1
    print(f"{n} ficheiros .txt gerados")

    secao("3. map_single_commit_to_graph (original), duas execuções")
    (PIPELINE / "function-data").mkdir()
    with open(PIPELINE / "function-data" / "linux-functions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File Path", "Vulnerable Commit Hash", "Vulnerable File Functions"])
        w.writerow([FICHEIRO_TESTE, COMMIT_TESTE, json.dumps(FUNCOES)])
    from convert_dot_to_json import map_single_commit_to_graph
    for execucao in (1, 2):
        with contextlib.redirect_stdout(io.StringIO()):
            map_single_commit_to_graph("linux", COMMIT_TESTE, TIPOS, "function-data/linux-functions.csv")
        print(f"execução {execucao}: " + ", ".join(
            f"{t}={sum(1 for _ in open(PIPELINE / 'output-data' / f'functions-{t}-linux.csv')) - 1} linhas" for t in TIPOS))

    secao("4. CSV -filtered no formato que create_embeddings.py lê")
    for tipo in TIPOS:
        origem = PIPELINE / "output-data" / f"functions-{tipo}-linux.csv"
        linhas = origem.read_text(encoding="utf-8").splitlines()
        # cabeçalho antigo (CFG_filepath) e separadores Linux, como nos CSV reais do repositório
        linhas[0] = linhas[0].replace("graph_filepath", "CFG_filepath")
        linhas = [linhas[0]] + [l.replace("\\", "/") for l in linhas[1:]]
        (PIPELINE / "output-data" / f"functions-{tipo}-linux-filtered.csv").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print("functions-{ast,cfg,pdg}-linux-filtered.csv criados")

    secao("5. create_embeddings.py (original) para cada tipo")
    shutil.copy(DADOS / "stub_transformers.py", PIPELINE / "transformers.py")
    original = (EXTRACAO / "create_embeddings.py").read_text(encoding="utf-8")
    for tipo in TIPOS:
        copia, n_subs = re.subn(r'^GRAPH_TYPES = \["pdg"\]', f'GRAPH_TYPES = ["{tipo}"]', original, flags=re.M)
        assert n_subs == 1, "linha GRAPH_TYPES não encontrada em create_embeddings.py"
        script = PIPELINE / f"create_embeddings_{tipo}.py"
        script.write_text(copia, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, script.name, "--csv", f"output-data/functions-{tipo}-linux-filtered.csv"],
                           cwd=PIPELINE, capture_output=True, text=True, encoding="utf-8", env=env)
        resumo = [l.strip() for l in r.stdout.splitlines() if re.search(r"Success|Skipped|Errors|Labels", l)]
        print(f"{tipo}: " + " | ".join(resumo))
        if r.returncode != 0:
            print(r.stderr[-3000:])
            sys.exit(1)

    print("\nPreparação concluída em", PIPELINE)


if __name__ == "__main__":
    main()
