"""O erro só troca os textos dos nós, ou também altera a estrutura do grafo?

Para cada grafo real da amostra (construir_amostra.py), constrói o grafo que a DGCNN recebe com o código original
(parse_dot_edges + create_dgl_graph_from_embedding) e verifica:
  1. estrutura: existe uma correspondência 1-para-1 entre os nós do grafo DGL e os nós do .dot que preserva
     TODAS as arestas (mesmo número de nós, mesmas arestas, incluindo repetidas). Se existe, os dois grafos são
     isomorfos, ou seja, têm exatamente a mesma estrutura;
  2. textos: em cada nó do grafo DGL, o texto (o vetor) é o do nó do .dot que ocupa essa posição?
  3. a correção proposta dá a estrutura certa E os textos certos.
"""
import contextlib
import io
import pickle
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from comum import CLASSIFICACAO, GERADO, TIPOS

import torch

sys.path.insert(0, str(CLASSIFICACAO))
from load_graphcodebert_embeddings import create_dgl_graph_from_embedding  # noqa: E402  (código original)

GRAFOS = GERADO / "impacto_tese" / "grafos"


def verificar(r):
    textos = dict(r["nos"])
    ordem = sorted(textos, key=int)
    # a feature de cada nó é o seu próprio id: assim sabe-se exatamente que nó o loader pôs em cada linha
    emb = {n: torch.tensor([float(i)], dtype=torch.float64) for i, n in enumerate(ordem)}
    d = {"embeddings": emb, "edge_index": torch.tensor(r["edge_index"], dtype=torch.long).reshape(2, -1),
         "node_to_idx": r["node_to_idx"]}
    with contextlib.redirect_stdout(io.StringIO()):
        g = create_dgl_graph_from_embedding(d)
    if g is None:
        return None
    conteudo = [ordem[int(v)] for v in g.ndata["features"][:, 0].tolist()]      # nó cujo texto está na linha i

    # papel estrutural de cada linha: o nó que parse_dot_edges numerou com i; as linhas a mais são nós sem arestas
    papel = [n for n, _ in sorted(r["node_to_idx"].items(), key=lambda kv: kv[1])]
    papel += sorted(set(textos) - set(papel), key=int)
    src, dst = g.edges()
    vistas = Counter((papel[s], papel[t]) for s, t in zip(src.tolist(), dst.tolist()))
    reais = Counter((a, b) for a, b in r["arestas"])
    bijecao = len(papel) == len(set(papel)) == g.num_nodes() == len(textos)
    mesma_estrutura = bijecao and vistas == reais

    # nós que entram em arestas mas não têm texto (linha vazia no .txt): o loader não lhes dá linha própria
    sem_texto = sum(1 for n in r["node_to_idx"] if n not in textos)
    tx = lambda n: textos.get(n, "<sem texto>")
    nos_texto_errado = sum(tx(p) != tx(c) for p, c in zip(papel, conteudo))
    pares_reais = Counter((tx(a), tx(b)) for a, b in r["arestas"])
    pares_vistos = Counter((tx(conteudo[s]), tx(conteudo[t])) for s, t in zip(src.tolist(), dst.tolist()))
    pares_certos = sum((pares_reais & pares_vistos).values())

    # correção: uma só ordem (numérica) para features e arestas
    idx = {n: i for i, n in enumerate(ordem)}
    corr_ok = all(a in idx and b in idx for a, b in r["arestas"])
    return {"nos": len(textos), "arestas": len(r["arestas"]), "mesma_estrutura": mesma_estrutura,
            "isolados": len(textos) - len(r["node_to_idx"]), "nos_texto_errado": nos_texto_errado,
            "pares_certos": pares_certos, "correcao_ok": corr_ok, "sem_texto": sem_texto}


def main():
    res = {t: [] for t in TIPOS}
    falhas = Counter()
    for p in sorted(GRAFOS.glob("*.pkl")):
        with open(p, "rb") as f:
            for r in pickle.load(f)["grafos"]:
                v = verificar(r)
                if v is None:
                    falhas[r["tipo"]] += 1
                else:
                    res[r["tipo"]].append(v)
    print(f"{len(list(GRAFOS.glob('*.pkl')))} ficheiros reais do kernel\n")
    print(f"{'tipo':4} {'grafos':>6} {'falham':>6} | {'estrutura igual':>16} | {'nós c/ texto errado':>20} {'arestas c/ par de textos certo':>31} | {'com nós isolados':>16}")
    for t in TIPOS:
        v = res[t]
        n = len(v)
        iguais = sum(x["mesma_estrutura"] for x in v)
        nos = sum(x["nos"] for x in v)
        errados = sum(x["nos_texto_errado"] for x in v)
        ar = sum(x["arestas"] for x in v)
        certos = sum(x["pares_certos"] for x in v)
        print(f"{t:4} {n:6} {falhas[t]:6} | {iguais:>6}/{n:<6} {iguais / n * 100:3.0f}% | "
              f"{errados:>8}/{nos:<7} {errados / nos * 100:3.0f}% {certos:>12}/{ar:<7} {certos / ar * 100:5.1f}%     | "
              f"{sum(x['isolados'] > 0 for x in v):>6}")
        diferentes = [x for x in v if not x["mesma_estrutura"]]
        if diferentes:
            com_sem_texto = sum(x["sem_texto"] > 0 for x in diferentes)
            print(f"     estrutura diferente em {len(diferentes)} grafos; destes, {com_sem_texto} têm nós sem texto nas "
                  f"arestas. Exemplo: {diferentes[0]}")
    todos = [x for t in TIPOS for x in res[t]]
    print(f"\nestrutura igual à do .dot em {sum(x['mesma_estrutura'] for x in todos)}/{len(todos)} grafos carregados")
    print("'falham' = o loader original não consegue construir o grafo (outro problema, fora deste teste)")
    print("'arestas c/ par de textos certo' conta pares (texto origem, texto destino) que existem de facto no .dot;"
          " inclui coincidências de textos iguais")


if __name__ == "__main__":
    main()
