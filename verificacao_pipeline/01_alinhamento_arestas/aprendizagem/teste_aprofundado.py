"""01 - Efeito da troca de ordem dos nós

Parte A - A estrutura do grafo mantém-se?
    Para os 18 grafos da preparação, compara o grafo DGL (loader original) com o .dot:
    forma do grafo (isomorfismo exato), conjunto de textos, quantos nós recebem o texto de outro nó,
    e se a troca é uma permutação aleatória ou um padrão fixo.

Parte B - Exemplo nó a nó (CFG de compute).

Parte C - Em que medida afeta o projeto: experiência controlada de aprendizagem.
    Gera funções C sintéticas, passa-as pelo Joern (CFG e PDG), constrói os grafos com o
    parse_dot_edges ORIGINAL + loader ORIGINAL (com o erro) e com a correção, e treina as classes
    ORIGINAIS DGCNNEncoder + DGCNNConv1DClassifier com os hiperparâmetros do script.
    Duas tarefas:
      - "conteúdo": a etiqueta depende só de QUE código existe (memcpy vs strncpy);
      - "estrutura": as duas classes têm exatamente as mesmas instruções, só muda a ORDEM
        (verificação do tamanho antes ou depois do memcpy). Só se resolve olhando para as arestas.
    Referência "sem arestas": o mesmo modelo, com o grafo sem arestas (só vê o conjunto de nós).
"""
import hashlib
import random
import statistics
import re
import sys
import types
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from comum import (CLASSIFICACAO, EXTRACAO, PIPELINE, TIPOS, atribuicoes_de_topo, cabecalho, campos_label,
                   carregar_definicoes, correr_joern, exigir_preparacao, ler_dot, secao, veredicto)

import dgl
import torch
import torch.nn as nn
sys.path.insert(0, str(CLASSIFICACAO))
from load_graphcodebert_embeddings import create_dgl_graph_from_embedding  # noqa: E402  (código original)

# create_embeddings.py importa o transformers ao nível do módulo; só precisamos de parse_dot_edges.
_stub = types.ModuleType("transformers")
_stub.RobertaTokenizer = _stub.RobertaModel = None
sys.modules.setdefault("transformers", _stub)
parse_dot_edges = carregar_definicoes(EXTRACAO / "create_embeddings.py", {"parse_dot_edges"})["parse_dot_edges"]

DV = CLASSIFICACAO / "detect_vulnerabilities.py"
CFG_DV = atribuicoes_de_topo(DV)
MODELO = carregar_definicoes(DV, {"DGCNNEncoder", "DGCNNConv1DClassifier"}, {"dropout_rate": CFG_DV["dropout_rate"]})


def texto(label_html):
    return campos_label(label_html)[-1]


def tipo_joern(nid, nomes):
    """Os ids do Joern codificam o tipo de nó nos bits altos (id >> 32)."""
    return nomes.get(int(nid) >> 32, f"tipo{int(nid) >> 32}")


def nomes_dos_tipos():
    """Na AST exportada, o 1.º campo da etiqueta é o tipo do nó (IDENTIFIER, METHOD, ...)."""
    nomes = Counter()
    for p in (PIPELINE / "output").rglob("*-ast.dot"):
        for nid, lab in ler_dot(p)[1].items():
            nomes[(int(nid) >> 32, campos_label(lab)[0].split(",")[0])] += 1
    out = {}
    for (cod, nome), _ in nomes.most_common():
        out.setdefault(cod, nome if nome.isupper() else "CALL")
    return out


# =========================================================================== Parte A + B
def parte_a_b():
    nomes = nomes_dos_tipos()
    secao("A. A estrutura do grafo mantém-se?")
    print(f"{'tipo':4} {'função':10} {'nós':>4} | {'forma igual':11} {'textos iguais':13} | "
          f"{'nós c/ texto de outro nó':>24} {'c/ texto de outro TIPO':>22} | padrão")
    tot_n = tot_errados = tot_tipo = 0
    todas_forma = todos_deslocamento = True
    exemplo = None
    for tipo in TIPOS:
        for pt in sorted((PIPELINE / "dataset_final_embeddings" / tipo).glob("*.pt")):
            d = torch.load(pt, weights_only=False)
            _, labels, arestas = ler_dot(PIPELINE / d["dot_path"])
            g = create_dgl_graph_from_embedding(d)

            # papel estrutural de cada linha i: o nó do .dot que o edge_index numerou com i (ordem de texto)
            papel = [n for n, _ in sorted(d["node_to_idx"].items(), key=lambda kv: kv[1])]
            # conteúdo de cada linha i: o vetor que o loader lá pôs (ordem numérica) - confirmado vetor a vetor
            feats = g.ndata["features"]
            conteudo = [None] * g.num_nodes()
            for nid, emb in d["embeddings"].items():
                (linha,) = (feats == emb.float()).all(dim=1).nonzero().flatten().tolist()
                conteudo[linha] = nid

            # forma: o grafo DGL, lido com os papéis estruturais, é EXATAMENTE o grafo do .dot?
            src, dst = g.edges()
            forma = Counter((papel[s], papel[t]) for s, t in zip(src.tolist(), dst.tolist())) == \
                Counter((a, b) for a, b, _ in arestas)
            textos = sorted(texto(labels[n]) for n in conteudo) == sorted(texto(l) for l in labels.values())
            errados = sum(p != c for p, c in zip(papel, conteudo))
            outro_tipo = sum(tipo_joern(p, nomes) != tipo_joern(c, nomes) for p, c in zip(papel, conteudo))
            # padrão: a linha i recebe o texto do nó que está m posições à frente na ordem dos ids,
            # sendo m o número de ids com 12 dígitos (METHOD, METHOD_RETURN, RETURN, PARAM, ...)
            m = sum(len(n) == 12 for n in papel)
            deslocado = conteudo == [papel[(i + m) % len(papel)] for i in range(len(papel))]
            todas_forma &= forma
            todos_deslocamento &= deslocado
            tot_n += len(papel)
            tot_errados += errados
            tot_tipo += outro_tipo
            print(f"{tipo:4} {d['function_name']:10} {len(papel):4} | {'sim' if forma else 'NÃO':11} "
                  f"{'sim' if textos else 'NÃO':13} | {errados:>4}/{len(papel):<3} ({errados / len(papel) * 100:3.0f}%)"
                  f"{'':8} {outro_tipo:>4} ({outro_tipo / len(papel) * 100:3.0f}%){'':9} | "
                  f"{'rotação de ' + str(m) if deslocado else 'outro'}")
            if tipo == "cfg" and d["function_name"] == "compute":
                exemplo = (papel, conteudo, labels, arestas, nomes, m)
    print(f"\ntotal: {tot_errados}/{tot_n} nós ({tot_errados / tot_n * 100:.0f}%) estão com o texto de outro nó; "
          f"{tot_tipo}/{tot_n} ({tot_tipo / tot_n * 100:.0f}%) com o texto de um nó de OUTRO TIPO")
    print(f"forma do grafo igual à do .dot em todos: {'sim' if todas_forma else 'NÃO'}; "
          f"troca é sempre uma rotação fixa: {'sim' if todos_deslocamento else 'NÃO'}")

    secao("B. Exemplo nó a nó: CFG de compute")
    papel, conteudo, labels, arestas, nomes, m = exemplo
    print(f"(m = {m} ids de 12 dígitos -> cada posição recebe o texto do nó {m} posições à frente)\n")
    print(f"{'pos':>3} | {'papel no grafo (nó real do .dot)':40} | {'texto que a GNN lá vê':28} | chega-lhe pelas arestas")
    pred = {}
    for a, b, _ in arestas:
        pred.setdefault(b, []).append(a)
    idx = {n: i for i, n in enumerate(papel)}
    for i, (p, c) in enumerate(zip(papel, conteudo)):
        real = f"{tipo_joern(p, nomes)}: {texto(labels[p])}"
        vindos = ", ".join(texto(labels[conteudo[idx[q]]]) for q in pred.get(p, [])) or "-"
        devia = ", ".join(texto(labels[q]) for q in pred.get(p, [])) or "-"
        print(f"{i:3} | {real[:40]:40} | {texto(labels[c])[:28]:28} | {vindos}   (devia ser: {devia})")
    print("\nA última coluna é o que a GNN agrega nesse nó (texto que lhe chega pelas arestas de entrada).")
    return todas_forma, tot_errados / tot_n


# =========================================================================== Parte C
FILLERS = ["t = t + 1;", "t = t * 3;", "t = n - t;", "t = t ^ 5;", "dst[1] = 0;", "t = t / 2 + n;",
           "if (t == 7) return 0;", "if (t < 0) return 1;"]   # os dois últimos acrescentam um RETURN (id de 12 dígitos)
EXTRA_PARAMS = ["int f1", "int f2", "long f3", "char f4"]       # cada parâmetro é um id de 12 dígitos no PDG


def corpo(rng, primeiro, segundo):
    """Instruções: fillers | primeiro | fillers | segundo | fillers."""
    f = lambda k: [rng.choice(FILLERS) for _ in range(rng.randint(0, k))]
    return f(3) + [primeiro] + f(2) + [segundo] + f(3)


def gerar_c(n_por_classe, seed=0):
    """Funções C sintéticas. Etiquetas em dois grupos de nomes: c_* (tarefa conteúdo), e_* (tarefa estrutura).
    O n.º de parâmetros e de returns varia ao acaso (independente da etiqueta), como nas funções reais:
    é isso que faz variar a rotação m de função para função."""
    rng = random.Random(seed)
    check = "if (n > 64) return -1;"
    funcs, etiquetas = [], {}
    for i in range(n_por_classe * 2):
        y = i % 2
        # conteúdo: verificação sempre antes; muda a função chamada
        chamada = "strncpy(dst, src, n);" if y == 0 else "memcpy(dst, src, n);"
        funcs.append(("c", i, y, corpo(rng, check, chamada), rng.randint(0, 4)))
        # estrutura: mesmas instruções nas duas classes; y=1 -> memcpy ANTES da verificação
        a, b = (check, "memcpy(dst, src, n);") if y == 0 else ("memcpy(dst, src, n);", check)
        funcs.append(("e", i, y, corpo(rng, a, b), rng.randint(0, 4)))
    linhas = ["#include <string.h>", ""]
    for tarefa, i, y, instr, n_extra in funcs:
        nome = f"{tarefa}_{i}"
        etiquetas[nome] = (tarefa, y)
        params = ", ".join(["char *dst", "const char *src", "int n"] + EXTRA_PARAMS[:n_extra])
        linhas.append(f"int {nome}({params})\n{{\n    int t = n;")
        linhas += [f"    {s}" for s in instr]
        linhas.append("    return t;\n}\n")
    return "\n".join(linhas), etiquetas


CLASSES_TEXTO = [("check", r"n > 64"), ("memcpy", r"memcpy"), ("strncpy", r"strncpy"), ("ret", r"return"),
                 ("cond", r"==|<"), ("atrib", r"="), ("método", r"^[ce]_\d+$")]


def classe_texto(s):
    for nome, padrao in CLASSES_TEXTO:
        if re.search(padrao, s):
            return nome
    return "outro"


def pares_vistos(g, textos_por_linha):
    """Saco de arestas tal como a GNN as vê: contagem de (classe do texto de origem, classe do texto de destino)."""
    src, dst = g.edges()
    return Counter((classe_texto(textos_por_linha[s]), classe_texto(textos_por_linha[t]))
                   for s, t in zip(src.tolist(), dst.tolist()))


_cache_tokens = {}


def embedding_texto(s, dim=768):
    """Substituto determinístico do GraphCodeBERT: soma de vetores aleatórios fixos por token."""
    v = torch.zeros(dim)
    for tok in re.findall(r"\w+|[^\w\s]", s):
        if tok not in _cache_tokens:
            gen = torch.Generator().manual_seed(int(hashlib.md5(tok.encode()).hexdigest()[:8], 16))
            _cache_tokens[tok] = torch.randn(dim, generator=gen)
        v += _cache_tokens[tok]
    return v / (v.norm() + 1e-8)


def grafos_de(pasta, etiquetas):
    """{tarefa: [(g_original, g_corrigido, g_sem_arestas, y)]}"""
    out = {"c": [], "e": []}
    for dot in sorted(Path(pasta).glob("*.dot")):
        nome, labels, arestas = ler_dot(dot)
        if nome not in etiquetas:
            continue
        tarefa, y = etiquetas[nome]
        emb = {n: embedding_texto(texto(l)) for n, l in labels.items()}
        edge_index, node_to_idx = parse_dot_edges(str(dot))                     # código original
        d = {"embeddings": emb, "edge_index": edge_index, "node_to_idx": node_to_idx}
        g_orig = create_dgl_graph_from_embedding(d)                              # loader original
        ordem = sorted(emb, key=int)                                             # correção: uma só ordem
        idx = {n: i for i, n in enumerate(ordem)}
        g_corr = dgl.graph(([idx[a] for a, _, _ in arestas], [idx[b] for _, b, _ in arestas]), num_nodes=len(ordem))
        g_corr.ndata["features"] = torch.stack([emb[n] for n in ordem]).float()
        g_vazio = dgl.graph(([], []), num_nodes=len(ordem))
        g_vazio.ndata["features"] = g_corr.ndata["features"]
        # texto em cada linha das features: em ambos os grafos é a ordem numérica (é o que o loader faz)
        textos = [texto(labels[n]) for n in ordem]
        sacos = (pares_vistos(g_orig, textos), pares_vistos(g_corr, textos), Counter(classe_texto(t) for t in textos))
        m = sum(len(n) == 12 for n in node_to_idx)
        out[tarefa].append((g_orig, g_corr, g_vazio, y, sacos, m))
    return out


def _matriz(sacos):
    from sklearn.feature_extraction import DictVectorizer
    chave = lambda k: "->".join(k) if isinstance(k, tuple) else k
    return DictVectorizer(sparse=False).fit_transform([{chave(k): v for k, v in s.items()} for s in sacos])


def sonda_linear(sacos, ys, treino=None):
    """Regressão logística sobre o saco de arestas. Mede se a informação está no grafo, sem depender da
    instabilidade do treino da DGCNN. Sem `treino`: validação cruzada 5x. Com `treino` (máscara booleana):
    treina nesses grafos e testa nos restantes."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    X, ys = _matriz(sacos), np.array(ys)
    if treino is None:
        return cross_val_score(LogisticRegression(max_iter=5000), X, ys, cv=5).mean()
    treino = np.array(treino)
    modelo = LogisticRegression(max_iter=5000).fit(X[treino], ys[treino])
    return modelo.score(X[~treino], ys[~treino])


def treinar_e_avaliar(treino, teste, seed):
    """Classes originais, hiperparâmetros do detect_vulnerabilities.py (Adam, lr, épocas, batch, CE)."""
    torch.manual_seed(seed)
    hd, k = CFG_DV["hidden_dimension"], CFG_DV["k_sortpooling"]
    enc = MODELO["DGCNNEncoder"](768, hd, k_sortpooling=k, pooling_type="sort_pooling", use_pooling=True)
    clf = MODELO["DGCNNConv1DClassifier"](input_dim=k * hd[-1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=CFG_DV["learning_rate"])
    perda = nn.CrossEntropyLoss()
    bs = CFG_DV["batch_size"]
    rng = random.Random(seed)
    enc.train(); clf.train()
    for _ in range(CFG_DV["num_epochs"]):
        ordem = list(range(len(treino)))
        rng.shuffle(ordem)
        for i in range(0, len(ordem), bs):
            lote = [treino[j] for j in ordem[i:i + bs]]
            out = clf(enc([g for g, _ in lote]))
            loss = perda(out, torch.tensor([y for _, y in lote]))
            opt.zero_grad(); loss.backward(); opt.step()
    enc.eval(); clf.eval()
    with torch.no_grad():
        pred = clf(enc([g for g, _ in teste])).argmax(1).tolist()
    return sum(p == y for p, (_, y) in zip(pred, teste)) / len(teste)


def dados_sinteticos(n_por_classe):
    codigo, etiquetas = gerar_c(n_por_classe)
    tmp_c = Path(__file__).resolve().parents[2] / "_gerado" / "aprofundamento_01"
    tmp_c.mkdir(parents=True, exist_ok=True)
    (tmp_c / "sintetico.c").write_text(codigo, encoding="utf-8")
    print(f"{len(etiquetas)} funções C sintéticas ({n_por_classe} por classe e por tarefa) -> Joern (cfg, pdg)")
    pastas = correr_joern(tmp_c / "sintetico.c", ["cfg", "pdg"])
    dados = {}
    for repr_, pasta in pastas.items():
        for tarefa, lista in grafos_de(pasta, etiquetas).items():
            random.Random(42).shuffle(lista)
            dados[(repr_, tarefa)] = lista
    ms = Counter(x[5] for x in dados[("pdg", "e")])
    print(f"rotação m nos PDG da tarefa estrutura (varia de função para função): {dict(sorted(ms.items()))}")
    return dados


NOMES_TAREFA = {"c": "conteúdo", "e": "estrutura"}


def parte_c1(dados):
    secao("C1. A informação estrutural chega à GNN? (sonda linear sobre o saco de arestas)")
    print("Cada grafo vira uma contagem de pares (texto de origem -> texto de destino) por aresta, tal como a")
    print("GNN os vê. 'sem arestas' = só a contagem de textos dos nós. Regressão logística, validação cruzada 5x.\n")
    print(f"{'grafo':5} {'tarefa':10} | {'sem erro (corrigido)':>20} {'com o erro (original)':>22} {'sem arestas':>12}")
    res = {}
    for (repr_, tarefa), lista in dados.items():
        ys = [x[3] for x in lista]
        acc = {nome: sonda_linear([x[4][i] for x in lista], ys) for i, nome in ((1, "corr"), (0, "orig"), (2, "vazio"))}
        res[(repr_, tarefa)] = acc
        print(f"{repr_:5} {NOMES_TAREFA[tarefa]:10} | {acc['corr'] * 100:19.1f}% {acc['orig'] * 100:21.1f}% "
              f"{acc['vazio'] * 100:11.1f}%")
    print("\n50% = acaso")

    secao("C1b. O que se aprende generaliza para funções com outra assinatura?")
    print("Treina nas funções com m pequeno (poucos parâmetros/returns) e testa nas de m grande.")
    print("O código relevante (verificação e memcpy) é o mesmo; só muda o número de parâmetros e de returns.\n")
    print(f"{'grafo':5} {'tarefa':10} | {'sem erro (corrigido)':>20} {'com o erro (original)':>22} | {'corte em m':>10}")
    for (repr_, tarefa), lista in dados.items():
        corte_m = statistics.median(x[5] for x in lista)
        treino = [x[5] <= corte_m for x in lista]
        ys = [x[3] for x in lista]
        acc = {nome: sonda_linear([x[4][i] for x in lista], ys, treino) for i, nome in ((1, "corr"), (0, "orig"))}
        res[(repr_, tarefa, "gen")] = acc
        print(f"{repr_:5} {NOMES_TAREFA[tarefa]:10} | {acc['corr'] * 100:19.1f}% {acc['orig'] * 100:21.1f}% | "
              f"treino m<={corte_m:g}")
    return res


def parte_c2(dados, seeds=(0, 1, 2, 3, 4)):
    import time
    secao("C2. A DGCNN original aprende a diferença?")
    print(f"modelo: DGCNNEncoder{tuple(CFG_DV['hidden_dimension'])} + SortPooling(k={CFG_DV['k_sortpooling']}) + "
          f"Conv1D; Adam lr={CFG_DV['learning_rate']}, {CFG_DV['num_epochs']} épocas, batch {CFG_DV['batch_size']}; "
          f"70/30 treino/teste; seeds {list(seeds)}\n")
    print(f"{'grafo':5} {'tarefa':10} | {'sem erro (corrigido)':>22} {'com o erro (original)':>22} {'sem arestas':>22}")
    res = {}
    for (repr_, tarefa), lista in dados.items():
        corte = int(0.7 * len(lista))
        t0 = time.time()
        acc = {}
        for col, nome in ((1, "corr"), (0, "orig"), (2, "vazio")):
            tr = [(x[col], x[3]) for x in lista[:corte]]
            te = [(x[col], x[3]) for x in lista[corte:]]
            acc[nome] = [treinar_e_avaliar(tr, te, s) for s in seeds]
        res[(repr_, tarefa)] = acc
        fmt = lambda v: f"{statistics.median(v) * 100:5.1f}% [{min(v) * 100:3.0f}-{max(v) * 100:3.0f}]"
        print(f"{repr_:5} {NOMES_TAREFA[tarefa]:10} | {fmt(acc['corr']):>22} {fmt(acc['orig']):>22} "
              f"{fmt(acc['vazio']):>22}   (n_teste={len(lista) - corte}, {time.time() - t0:.0f}s)")
    print("\nvalores: mediana da exatidão no teste [mín-máx entre seeds]; 50% = acaso")
    return res


def main():
    cabecalho("01 - Efeito da troca de ordem dos nós")
    exigir_preparacao()
    forma, frac_errados = parte_a_b()
    secao("C. Em que medida afeta o projeto: experiência controlada")
    dados = dados_sinteticos(n_por_classe=150)
    sonda = parte_c1(dados)
    if "--sem-dgcnn" not in sys.argv:
        parte_c2(dados)

    secao("Conclusão")
    print(f"- forma do grafo preservada: {'sim' if forma else 'não'}; mas {frac_errados * 100:.0f}% dos nós têm o texto "
          "de outro nó (rotação pelos ids de 12 dígitos, diferente em cada função).")
    for r in ("cfg", "pdg"):
        c, e = sonda[(r, "c")], sonda[(r, "e")]
        print(f"- {r}: conteúdo {c['corr'] * 100:.0f}% vs {c['orig'] * 100:.0f}% (corrigido vs erro); "
              f"estrutura {e['corr'] * 100:.0f}% vs {e['orig'] * 100:.0f}% (sem arestas: {e['vazio'] * 100:.0f}%); "
              f"estrutura noutra assinatura {sonda[(r, 'e', 'gen')]['corr'] * 100:.0f}% vs "
              f"{sonda[(r, 'e', 'gen')]['orig'] * 100:.0f}%")
    veredicto(True, "CRÍTICA",
              "a forma do grafo mantém-se, mas os textos rodam de posição; a GNN continua a ver QUE código existe, "
              "mas o que vê sobre COMO está ligado deixa de corresponder ao código.")


if __name__ == "__main__":
    main()
