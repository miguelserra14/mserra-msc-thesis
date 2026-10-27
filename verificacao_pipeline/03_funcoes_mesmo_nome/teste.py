"""03 - Duas funções com o mesmo nome no mesmo ficheiro ficam com o grafo e a etiqueta certos?

map_single_commit_to_graph faz um dicionário nome -> .dot; com nomes repetidos (p. ex. os dois ramos
de um #ifdef) ambas as funções apontam para o mesmo .dot e create_embeddings.py guarda a etiqueta da
primeira linha. Mede o efeito no teste e nos dados reais do Linux e demonstra o mapeamento por nome + linha.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (PIPELINE, cabecalho, campos_label, exigir_preparacao, ler_dot, ler_funcoes_fonte, nomes_dos_dots,
                   pasta_pipeline, secao, veredicto)

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_preparacao"))
from preparar import FUNCOES  # noqa: E402  (lista de funções usada no CSV de teste)


def linha_do_metodo(caminho_dot):
    _, labels, _ = ler_dot(caminho_dot)
    for label in labels.values():
        campos = campos_label(label)
        m = re.match(r"METHOD, (\d+)$", campos[0])
        if m:
            return int(m.group(1))
    return None


def main():
    cabecalho("03 - Funções com o mesmo nome no mesmo ficheiro")
    exigir_preparacao()

    secao("No ficheiro de teste (pipeline original)")
    pasta = pasta_pipeline("pdg")
    dots_helper = [f for f, n in nomes_dos_dots(pasta).items() if n == "helper"]
    print(f".dot exportados com o nome 'helper': {dots_helper}  (linhas {[linha_do_metodo(pasta / f) for f in dots_helper]})")
    m = pd.read_csv(PIPELINE / "output-data" / "functions-pdg-linux-filtered.csv", sep=";").drop_duplicates()
    hm = m[m.function_name == "helper"][["function_name", "CFG_filepath", "vulnerable_label"]]
    hm = hm.assign(ficheiro_dot=hm.CFG_filepath.str.split("/").str[-1]).drop(columns="CFG_filepath")
    print("linhas do CSV de mapeamento:\n" + hm.to_string(index=False))
    usados = set(hm.ficheiro_dot)
    print(f".dot de 'helper' nunca usado: {sorted(set(dots_helper) - usados)}")
    pts = {p.name.split("__")[-1]: torch.load(p, weights_only=False)
           for p in (PIPELINE / "dataset_final_embeddings" / "pdg").glob("*.pt")}
    etiqueta_pt = {k: int(v["label"]) for k, v in pts.items() if v["function_name"] == "helper"}
    print(f"etiqueta guardada no .pt: {etiqueta_pt}  (a 'helper' vulnerável perdeu a etiqueta 1)")
    perdeu = 1 not in etiqueta_pt.values()

    secao("Correção demonstrada: mapear por nome + linha de início")
    linhas_dots = {f: (n, linha_do_metodo(pasta / f)) for f, n in nomes_dos_dots(pasta).items()}
    for fn in FUNCOES:
        l0, l1 = fn["Lines"]
        candidatos = [f for f, (n, l) in linhas_dots.items() if n == fn["Name"] and l is not None and l0 <= l <= l1]
        print(f"{fn['Name']:10} linhas {l0:>2}-{l1:<2} vulnerável={fn['Vulnerable']:3} -> {candidatos}")

    secao("Nos dados reais do Linux (linux-functions.csv)")
    fn = ler_funcoes_fonte()
    dup = fn[fn.duplicated(["commit", "file", "name"], keep=False)]
    grupos = dup.groupby(["commit", "file", "name"]).vuln.agg(["min", "max"])
    conflitos = int((grupos["min"] != grupos["max"]).sum())
    print(f"funções que partilham (commit, ficheiro, nome) com outra: {len(dup)} de {len(fn)} "
          f"({len(dup) / len(fn) * 100:.2f}%), das quais vulneráveis: {int(dup.vuln.sum())}")
    print(f"grupos com etiquetas diferentes (um grafo fica com a etiqueta errada): {conflitos}")

    veredicto(perdeu, "BAIXA",
              f"o erro existe e foi reproduzido (a 'helper' vulnerável ficou com etiqueta 0), mas nos dados do Linux "
              f"só afeta {len(dup)} funções e {conflitos} conflito de etiqueta. Mapear por nome + linha resolve.")


if __name__ == "__main__":
    main()
