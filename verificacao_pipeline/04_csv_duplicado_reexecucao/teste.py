"""04 - Voltar a correr a extração duplica o CSV de mapeamento?

map_single_commit_to_graph abre o CSV em modo 'a' (acrescentar). A preparação correu-o duas vezes
sobre o mesmo commit, tal como acontece quando main.py é relançado. Mede também o efeito nos CSV reais.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import EXTRACAO, PIPELINE, TIPOS, cabecalho, exigir_preparacao, ler_mapeamento, secao, veredicto

import pandas as pd


def main():
    cabecalho("04 - Linhas duplicadas no CSV de mapeamento ao reexecutar")
    exigir_preparacao()

    secao("No teste: duas execuções de map_single_commit_to_graph (original)")
    duplicado_no_teste = False
    for tipo in TIPOS:
        m = pd.read_csv(PIPELINE / "output-data" / f"functions-{tipo}-linux.csv", sep=";")
        print(f"{tipo}: {len(m)} linhas, {len(m.drop_duplicates())} únicas")
        duplicado_no_teste |= len(m) > len(m.drop_duplicates())
    n_pt = len(list((PIPELINE / "dataset_final_embeddings" / "pdg").glob("*.pt")))
    print(f".pt gerados para pdg: {n_pt} (create_embeddings.py ignora caminhos já processados, por isso não há .pt duplicados)")

    secao("Nos CSV reais do repositório (output-data/)")
    print(f"{'ficheiro':38} {'linhas':>8} {'únicas':>8} {'duplicadas':>10}")
    for tipo in TIPOS:
        for filtrado in (False, True):
            m = ler_mapeamento(tipo, filtrado)
            u = len(m.drop_duplicates())
            nome = f"functions-{tipo}-linux{'-filtered' if filtrado else ''}.csv"
            print(f"{nome:38} {len(m):8} {u:8} {len(m) - u:10}")
    nao_lidos = sorted(p.name for p in (EXTRACAO / "output-data").glob("functions-linux-*.csv"))
    print(f"\nficheiros escritos por map_functions_to_graph que nenhum script lê: {nao_lidos}")

    secao("Diferença entre o CSV original e o -filtered (o 'filtro' não tem script no repositório)")
    for tipo in TIPOS:
        a = set(map(tuple, ler_mapeamento(tipo, False).drop_duplicates().astype(str).values))
        b = set(map(tuple, ler_mapeamento(tipo, True).drop_duplicates().astype(str).values))
        print(f"{tipo}: linhas únicas só no original = {len(a - b)}, só no -filtered = {len(b - a)}")

    veredicto(duplicado_no_teste, "BAIXA",
              "cada reexecução acrescenta outra vez todas as linhas (os CSV reais têm entre 26% e 45% de linhas duplicadas). "
              "Sozinho não muda os dados (create_embeddings.py salta caminhos repetidos), mas a primeira linha "
              "decide a etiqueta e permite que linhas antigas e novas se misturem (ver teste 05).")


if __name__ == "__main__":
    main()
