"""13 - Que informação do grafo e do código chega de facto à GNN?

(a) tipos de aresta do PDG (CDG = controlo, DDG = dados) e arestas repetidas;
(b) sentido da propagação: o GraphConv do DGL só agrega dos vizinhos de ENTRADA;
(c) o GraphCodeBERT recebe cada nó sozinho, sem o grafo de data-flow que o distingue do CodeBERT.
"""
import ast
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (EXTRACAO, PIPELINE, cabecalho, campos_label, exigir_preparacao, ler_ast, ler_dot, pasta_pipeline,
                   secao, veredicto)

import dgl
import torch
from dgl.nn.pytorch import GraphConv


def main():
    cabecalho("13 - Representação do grafo que chega à GNN")
    exigir_preparacao()

    secao("(a) Tipos de aresta do PDG e arestas repetidas")
    tipos, repetidas, total = Counter(), 0, 0
    for dot in pasta_pipeline("pdg").glob("*.dot"):
        nome, _, arestas = ler_dot(dot)
        if nome not in {"copy_item", "helper", "log_msg", "compute", "get_len", "main"}:
            continue
        tipos.update(re.match(r"(\w+)", lab).group(1) if lab else "?" for _, _, lab in arestas)
        pares = Counter((a, b) for a, b, _ in arestas)
        repetidas += sum(c - 1 for c in pares.values() if c > 1)
        total += len(arestas)
    pt = torch.load(next((PIPELINE / "dataset_final_embeddings" / "pdg").glob("*.pt")), weights_only=False)
    print(f"arestas nos .dot do PDG por tipo: {dict(tipos)}")
    print(f"chaves guardadas no .pt: {sorted(pt.keys())}  -> nenhuma guarda o tipo de aresta")
    print(f"arestas repetidas (mesmo par de nós, variáveis diferentes): {repetidas} de {total}; "
          "no DGL viram arestas múltiplas, que o GraphConv conta mais do que uma vez")

    secao("(b) Sentido da propagação no GraphConv (DGL)")
    g = dgl.add_self_loop(dgl.graph(([0], [1]), num_nodes=2))     # aresta 0 -> 1
    conv = GraphConv(1, 1, allow_zero_in_degree=True, bias=False)
    x1 = torch.tensor([[1.0], [0.0]])
    x2 = torch.tensor([[1.0], [5.0]])                               # só muda o nó 1
    print(f"grafo 0 -> 1: mudar o nó 1 altera a saída do nó 0? {not torch.allclose(conv(g, x1)[0], conv(g, x2)[0])}")
    _, labels, arestas = ler_dot(pasta_pipeline("ast") / "5-ast.dot")          # AST de compute
    raiz = next(n for n, l in labels.items() if campos_label(l)[0].startswith("METHOD,"))
    entrada = sum(1 for a, b, _ in arestas if b == raiz)
    print(f"AST de compute: arestas a entrar no nó METHOD (raiz) = {entrada} -> a raiz só se vê a si própria; "
          "a informação desce da raiz para as folhas e nunca sobe")

    secao("(c) Entrada do GraphCodeBERT em create_embeddings.py")
    arvore = ler_ast(EXTRACAO / "create_embeddings.py")
    for n in ast.walk(arvore):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in {"tokenizer", "model"}:
            print(f"linha {n.lineno}: {ast.unparse(n)[:110]}")
    print("cada nó é tokenizado sozinho (lista de textos de nós) e o modelo só recebe input_ids e attention_mask;")
    print("não há position_idx nem a máscara de data-flow que o GraphCodeBERT usa para ler o grafo de variáveis.")

    veredicto(True, "BAIXA",
              "limitações de desenho, não erros: perde-se o tipo de aresta do PDG, a informação na AST só flui da raiz "
              "para as folhas, e o GraphCodeBERT é usado como um codificador de texto por nó (sem o data-flow que o "
              "relatório lhe atribui).")


if __name__ == "__main__":
    main()
