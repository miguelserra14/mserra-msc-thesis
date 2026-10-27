"""01 - As arestas que a DGCNN recebe ligam os nós certos?

Carrega os .pt gerados pelo create_embeddings.py ORIGINAL com o loader ORIGINAL do treino
(load_graphcodebert_embeddings.create_dgl_graph_from_embedding) e compara, aresta a aresta,
o grafo DGL com as arestas do .dot do Joern (lidas com um parser independente).
Depois aplica a correção proposta e repete a comparação.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (CLASSIFICACAO, PIPELINE, TIPOS, cabecalho, campos_label, exigir_preparacao, ler_dot, secao,
                   veredicto)

import dgl
import torch

sys.path.insert(0, str(CLASSIFICACAO))
from load_graphcodebert_embeddings import create_dgl_graph_from_embedding  # noqa: E402  (código original)


def ids_por_linha(g, d):
    """Descobre que nó do Joern ocupa cada linha de g.ndata['features'] (comparação exata de vetores)."""
    feats = g.ndata["features"]
    linha_para_id = {}
    for nid, emb in d["embeddings"].items():
        linhas = (feats == emb.float()).all(dim=1).nonzero().flatten().tolist()
        assert len(linhas) == 1, "vetor não encontrado ou repetido"
        linha_para_id[linhas[0]] = nid
    return linha_para_id


def arestas_vistas_pela_gnn(g, linha_para_id):
    src, dst = g.edges()
    return [(linha_para_id[s], linha_para_id[t]) for s, t in zip(src.tolist(), dst.tolist())]


def grafo_corrigido(d, caminho_dot):
    """Correção: uma única ordem de nós (a do loader: id numérico) usada para features E arestas."""
    ordem = sorted(d["embeddings"], key=int)
    idx = {nid: i for i, nid in enumerate(ordem)}
    _, _, arestas = ler_dot(caminho_dot)
    src = [idx[a] for a, b, _ in arestas]
    dst = [idx[b] for a, b, _ in arestas]
    g = dgl.graph((src, dst), num_nodes=len(ordem))
    g.ndata["features"] = torch.stack([d["embeddings"][n] for n in ordem]).float()
    return g


def texto(label_html, n=22):
    return campos_label(label_html)[-1][:n]


def main():
    cabecalho("01 - Alinhamento entre features dos nós e arestas (edge_index)")
    exigir_preparacao()
    total = certas = certas_corr = 0
    exemplo = None
    comprimentos_ids = set()
    isolados = 0

    secao("Grafos produzidos pela pipeline original")
    print(f"{'tipo':4} {'função':10} {'nós':>4} {'arestas':>7} {'certas':>6} {'%':>5} | após correção")
    for tipo in TIPOS:
        for pt in sorted((PIPELINE / "dataset_final_embeddings" / tipo).glob("*.pt")):
            d = torch.load(pt, weights_only=False)
            caminho_dot = PIPELINE / d["dot_path"]
            _, labels, arestas_dot = ler_dot(caminho_dot)
            verdade = {(a, b) for a, b, _ in arestas_dot}
            comprimentos_ids |= {len(n) for n in labels}
            isolados += len(set(labels) - {a for a, _, _ in arestas_dot} - {b for _, b, _ in arestas_dot})

            g = create_dgl_graph_from_embedding(d)          # loader original
            vistas = arestas_vistas_pela_gnn(g, ids_por_linha(g, d))
            ok = sum(1 for e in vistas if e in verdade)

            gc = grafo_corrigido(d, caminho_dot)
            vistas_c = arestas_vistas_pela_gnn(gc, ids_por_linha(gc, d))
            ok_c = sum(1 for e in vistas_c if e in verdade)

            total += len(vistas)
            certas += ok
            certas_corr += ok_c
            print(f"{tipo:4} {d['function_name']:10} {g.num_nodes():4} {len(vistas):7} {ok:6} {ok / len(vistas) * 100:4.0f}% | {ok_c}/{len(vistas_c)}")
            if tipo == "cfg" and d["function_name"] == "compute":
                inv = {v: k for k, v in d["node_to_idx"].items()}
                src, dst = g.edges()
                exemplo = [(texto(labels[inv[s]]), texto(labels[inv[t]]), texto(labels[a]), texto(labels[b]))
                           for (s, t), (a, b) in zip(zip(src.tolist(), dst.tolist()), vistas)]

    secao("Exemplo: CFG de compute")
    print(f"{'aresta real (.dot)':47} | aresta que a GNN recebe")
    for a, b, c, e in exemplo:
        print(f"{a:22} -> {b:22} | {c:22} -> {e}")

    secao("Porque acontece")
    print("create_embeddings.py:130  node_to_idx = ordem de TEXTO dos ids  (sorted(set(...)))")
    print("load_graphcodebert_embeddings.py:43  features por ordem NUMÉRICA (key=int)")
    print(f"comprimentos dos ids do Joern nestes grafos: {sorted(comprimentos_ids)} dígitos "
          "-> com ids de 11 e 12 dígitos no mesmo grafo as duas ordenações divergem (só coincidem por acaso)")
    print(f"nós sem nenhuma aresta nestes grafos: {isolados} (cada um deslocaria ainda mais a numeração do edge_index)")

    secao("Totais")
    print(f"pipeline original: {certas}/{total} arestas corretas ({certas / total * 100:.1f}%)")
    print(f"com a correção:    {certas_corr}/{total} arestas corretas ({certas_corr / total * 100:.1f}%)")

    veredicto(certas < total, "CRÍTICA",
              f"só {certas / total * 100:.0f}% das arestas ligam os nós certos; a topologia mantém-se mas o conteúdo "
              f"dos nós fica trocado. A correção recupera {certas_corr}/{total}.")


if __name__ == "__main__":
    main()
