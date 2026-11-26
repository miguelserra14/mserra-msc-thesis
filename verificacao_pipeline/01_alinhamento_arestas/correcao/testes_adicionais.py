"""01 - Testes adicionais da correção (complementam testar_correcao.py).

  A1  Testes de mutação: estraga-se a correção de 7 maneiras plausíveis; a bateria de testes tem de apanhar todas.
      Mostra que os testes têm poder para detetar uma correção errada (não passam "por acaso"). Depois, 7 mutantes do
      embedding_dataset.py (alinhamento, filtro dos grafos, interruptor), com uma bateria própria.
  A2  Fuzzing: 20 000 grafos aleatórios (ids de 1 a 19 dígitos, arestas repetidas, laços, nós isolados, nós sem texto),
      .pt antigos e novos. O loader corrigido tem de dar exatamente as arestas do .dot, ou descartar o grafo
      quando (e só quando) uma aresta toca num nó sem texto.
  A3  Joern com código adversarial: strings e comentários com '"123" -> "456"'. Compara 3 leitores de arestas
      (regex da Constança, parser de linhas de comum.py, pydot) em todos os .dot gerados e nos 75 .dot do 00_preparacao.
  A4  Vetores reais do GraphCodeBERT (vetores.npy) nos 6659 grafos da amostra: identifica-se o nó de cada linha
      pelo VETOR (não pelo id) e compara-se cada aresta, como par de textos, com o .dot.
  A5  Caminhos que contornam a correção: outros leitores de .pt (os que usam o edge_index gravado têm de aplicar a
      correção e estar testados; os que reconstroem as arestas a partir do .dot são verificados, como o
      fusion_all_graphs.py) e caminhos de runs fixos no código.

Uso (a partir de verificacao_pipeline):
  python -X utf8 01_alinhamento_arestas/correcao/testes_adicionais.py
"""
import inspect
import pickle
import random
import re
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import testar_correcao as TC  # noqa: E402  (reutiliza os módulos e utilitários; não corre o main)
from comum import CLASSIFICACAO, DADOS, GERADO, PIPELINE, cabecalho, correr_joern, ler_dot, secao  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CORRIGIDO = TC.CORRIGIDO
verificar, falhas = TC.verificar, TC.falhas


def registos_amostra(limite=None):
    n = 0
    for p in sorted(TC.GRAFOS.glob("*.pkl")):
        with open(p, "rb") as f:
            for r in pickle.load(f)["grafos"]:
                yield r
                n += 1
                if limite and n >= limite:
                    return


def pt_de_registo(r, emb):
    return {"embeddings": emb, "edge_index": torch.tensor(r["edge_index"], dtype=torch.long).reshape(2, -1),
            "node_to_idx": r["node_to_idx"]}


# --------------------------------------------------------------------------- A1
MUTANTES = {
    "sem tradução (idx_to_row = identidade, i.e. o erro original)":
        ("idx_to_row[idx] = row.get(node, -1)", "idx_to_row[idx] = idx"),
    "linhas por ordem de texto em vez de numérica":
        ("row = {node: i for i, node in enumerate(node_ids)}", "row = {node: i for i, node in enumerate(sorted(node_ids))}"),
    "origem e destino trocados":
        ("aligned = idx_to_row[edge_index.long()]", "aligned = idx_to_row[edge_index.long().flip(0)]"),
    "nó sem texto aceite em silêncio (sem a verificação final)":
        ("if (aligned < 0).any():", "if False:"),
    "nó sem texto mandado para a linha 0":
        ("idx_to_row[idx] = row.get(node, -1)", "idx_to_row[idx] = row.get(node, 0)"),
    "node_to_idx com índices repetidos aceite":
        ("if sorted(node_to_idx.values()) != list(range(len(node_to_idx))):", "if False:"),
    "edge_index sem node_to_idx aceite (usa-o tal como está)":
        ('raise ValueError("edge_index without node_to_idx: cannot align features with edges")', "return edge_index"),
}


def bateria(n_reais=400):
    """Os testes que devem matar qualquer mutante: casos sintéticos do T1 + n_reais grafos reais (T3).
    Devolve a lista de verificações que falharam."""
    falhou = []
    M, R, C1, C2, I, ISO = "107374182400", "128849018880", "30064771072", "30064771073", "68719476736", "68719476737"
    arestas = [(M, C1), (C1, C2), (C2, I), (I, R), (M, R), (C1, C2)]
    emb, ordem = TC.emb_codificado([M, R, C1, C2, I, ISO])
    with tempfile.TemporaryDirectory() as tmp:
        dot, dot_p = Path(tmp) / "f.dot", Path(tmp) / "p.dot"
        TC.escrever_dot(dot, arestas)
        TC.escrever_dot(dot_p, arestas + [(C2, "30064771099")])
        ei, n2i = TC.parse_original(str(dot))
        antigo = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
        ei, n2i = TC.parse_original(str(dot_p))
        antigo_p = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
    g = TC.g_corrigido(antigo)
    if g is None or TC.arestas_por_id(g, ordem) != Counter(arestas):
        falhou.append("sintético: arestas")
    if g is not None and list(zip(*[x.tolist() for x in g.edges()])) != [(ordem.index(a), ordem.index(b)) for a, b in arestas]:
        falhou.append("sintético: ordem/direção das arestas")
    if TC.g_corrigido(antigo_p) is not None:
        falhou.append("sintético: nó sem texto")
    if TC.g_corrigido({"embeddings": emb, "edge_index": antigo["edge_index"]}) is not None:
        falhou.append("sintético: sem node_to_idx")
    # índice repetido sem que falte nenhum índice usado pelas arestas: sem a verificação, um dos nós ganharia em silêncio
    dup = {"embeddings": emb, "edge_index": torch.tensor([[0, 1], [1, 3]]), "node_to_idx": {M: 0, C1: 1, C2: 1, R: 3}}
    if TC.g_corrigido(dup) is not None:
        falhou.append("sintético: node_to_idx repetido")
    for r in registos_amostra(n_reais):
        textos = dict(r["nos"])
        emb, ordem = TC.emb_codificado(textos)
        gc = TC.g_corrigido(pt_de_registo(r, emb))
        sem_texto = any(a not in textos or b not in textos for a, b in r["arestas"])
        if sem_texto != (gc is None) or (gc is not None and TC.arestas_por_id(gc, ordem) != Counter(map(tuple, r["arestas"]))):
            falhou.append(f"real: {r['tipo']} {r['funcao']}")
            break
    return falhou


def a1():
    secao("A1  Testes de mutação: a bateria tem de detetar cada versão estragada da correção")
    fonte = textwrap.dedent(inspect.getsource(CORRIGIDO.aligned_edge_index))
    verdadeira = CORRIGIDO.aligned_edge_index
    verificar(not bateria(), "a correção verdadeira passa a bateria (sem isto, matar mutantes não prova nada)")
    for nome, (antes, depois) in MUTANTES.items():
        if antes not in fonte:
            verificar(False, f"mutante '{nome}': texto a mutar não encontrado")
            continue
        ns = dict(vars(CORRIGIDO))
        exec(compile(fonte.replace(antes, depois, 1), "<mutante>", "exec"), ns)
        CORRIGIDO.aligned_edge_index = ns["aligned_edge_index"]
        try:
            apanhado = bateria()
        finally:
            CORRIGIDO.aligned_edge_index = verdadeira
        verificar(bool(apanhado), f"mutante apanhado: {nome}  <- {apanhado[0] if apanhado else 'NÃO DETETADO'}")
    a1_embedding_dataset()


# mutantes do embedding_dataset.py (os de cima mutam a aligned_edge_index, que ele importa do loader)
MUTANTES_ED = {
    "EmbeddingDataset: get() não alinha as arestas":
        ("                edge_index = aligned_edge_index(data_dict, node_ids)", "                pass"),
    "EmbeddingDataset: get() alinha com as linhas por ordem de texto":
        ("edge_index = aligned_edge_index(data_dict, node_ids)", "edge_index = aligned_edge_index(data_dict, sorted(node_ids))"),
    "EmbeddingDataset: __init__ não tira os grafos que não se podem alinhar":
        ("            self.file_paths = self._alignable_files(self.file_paths)", "            pass"),
    "EmbeddingDataset: o filtro não verifica o alinhamento":
        ("                aligned_edge_index(data_dict, node_ids)\n            except", "                pass\n            except"),
    "EmbeddingDataset: o filtro não verifica que as features empilham (grafo sem nós)":
        ("                torch.stack([embeddings_dict[node_id] for node_id in node_ids])\n", ""),
    "EmbeddingDataset: o modo original também alinha (ignora o interruptor no get())":
        ("if not LEGACY_EDGE_ALIGNMENT:  # [fix 01] map the edges to the rows of x", "if True:"),
    "EmbeddingDataset: o modo original também filtra (ignora o interruptor no __init__)":
        ("if not LEGACY_EDGE_ALIGNMENT:  # [fix 01] keep only", "if True:  # keep only"),
}


def modulo_ed(fonte):
    """Executa o código-fonte de um embedding_dataset.py (verdadeiro ou mutante) num módulo novo."""
    import types
    m = types.ModuleType("ed_mutante")
    m.__file__ = str(CLASSIFICACAO / "embedding_dataset.py")
    exec(compile(fonte, m.__file__, "exec"), m.__dict__)
    return m


def bateria_ed(m, n_reais=300):
    """Os testes que devem matar qualquer mutante do embedding_dataset.py: casos sintéticos (normal, nó sem texto,
    sem nós, sem node_to_idx) + n_reais grafos reais. No modo corrigido tem de dar os grafos do loader DGL corrigido;
    no original, o código da Constança. Devolve a lista do que falhou."""
    falhou = []
    M, R, C1, C2, I, ISO = "107374182400", "128849018880", "30064771072", "30064771073", "68719476736", "68719476737"
    arestas = [(M, C1), (C1, C2), (C2, I), (I, R), (M, R), (C1, C2)]
    emb, _ = TC.emb_codificado([M, R, C1, C2, I, ISO])
    with tempfile.TemporaryDirectory() as tmp:
        raiz = Path(tmp)
        TC.escrever_pts(registos_amostra(n_reais), raiz)
        dot, dot_p = raiz / "f.dot", raiz / "p.dot"
        TC.escrever_dot(dot, arestas)
        TC.escrever_dot(dot_p, arestas + [(C2, "30064771099")])
        sinteticos = {"normal": dict(zip(("edge_index", "node_to_idx"), TC.parse_original(str(dot))), embeddings=emb),
                      "sem_texto": dict(zip(("edge_index", "node_to_idx"), TC.parse_original(str(dot_p))), embeddings=emb),
                      "sem_nos": {"embeddings": {}, "edge_index": TC.VAZIO, "node_to_idx": {}}}
        sinteticos["sem_node_to_idx"] = {"embeddings": emb, "edge_index": sinteticos["normal"]["edge_index"]}
        (raiz / "ast").mkdir(exist_ok=True)
        for nome, d in sinteticos.items():
            torch.save(dict(d, label=torch.tensor(1), function_name=nome, graph_type="ast"), raiz / "ast" / f"s_{nome}.pt")
        dgl_c = {p: TC.g_corrigido(torch.load(p, weights_only=False)) for t in TC.TIPOS for p in (raiz / t).glob("*.pt")}
        esperados = {p for p, g in dgl_c.items() if g is not None}
        fix, _ = TC.ed_ler(m, raiz, legacy=False)
        if set(fix) != esperados:
            falhou.append("corrigido: grafos carregados")
        elif not all(TC.data_igual_dgl(fix[p], dgl_c[p]) for p in esperados):
            falhou.append("corrigido: x/edge_index")
        leg, s_leg = TC.ed_ler(m, raiz, legacy=True)
        orig, s_orig = TC.ed_ler(TC.ed_original(), raiz)
        if list(leg) != list(orig) or s_leg != s_orig or not all(TC.mesmo_data(leg[p], orig[p]) for p in orig):
            falhou.append("modo original != código original")
    return falhou


def a1_embedding_dataset():
    fonte = (CLASSIFICACAO / "embedding_dataset.py").read_text(encoding="utf-8")
    verificar(not bateria_ed(modulo_ed(fonte)), "EmbeddingDataset: o código verdadeiro passa a bateria")
    for nome, (antes, depois) in MUTANTES_ED.items():
        if fonte.count(antes) != 1:
            verificar(False, f"mutante '{nome}': texto a mutar não encontrado (ou repetido)")
            continue
        apanhado = bateria_ed(modulo_ed(fonte.replace(antes, depois, 1)))
        verificar(bool(apanhado), f"mutante apanhado: {nome}  <- {apanhado[0] if apanhado else 'NÃO DETETADO'}")


# --------------------------------------------------------------------------- A2
def id_aleatorio(rng, usados):
    while True:
        d = rng.choice([1, 2, 5, 10, 11, 11, 11, 12, 12, 13, 19])
        x = str(rng.randrange(10 ** (d - 1) if d > 1 else 0, 10 ** d))
        if x not in usados:
            usados.add(x)
            return x


def a2(n=20000, seed=1234):
    secao(f"A2  Fuzzing: {n} grafos aleatórios, .pt antigo e novo")
    rng = random.Random(seed)
    c = Counter()
    with tempfile.TemporaryDirectory() as tmp:
        dot = Path(tmp) / "g.dot"
        for _ in range(n):
            usados = set()
            nos = [id_aleatorio(rng, usados) for _ in range(rng.randint(1, 40))]
            p_sem = rng.choice([0, 0, 0, 0.05, 0.3])
            com_texto = [x for x in nos if rng.random() >= p_sem] or nos[:1]
            arestas = [(rng.choice(nos), rng.choice(nos)) for _ in range(rng.choice([0, 1, rng.randint(1, 120)]))]
            TC.escrever_dot(dot, arestas)
            emb, ordem = TC.emb_codificado(com_texto)
            ei, n2i = TC.parse_original(str(dot))
            antigo = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
            ei, n2i = TC.parse_corrigido(str(dot), emb.keys())
            novo = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
            ga, gn, go = TC.g_corrigido(antigo), TC.g_corrigido(novo), TC.g_original(antigo)
            descartar = any(a not in emb or b not in emb for a, b in arestas)
            c["grafos"] += 1
            c["descartar_esperado"] += descartar
            for nome, g in (("antigo", ga), ("novo", gn)):
                if descartar:
                    c[f"ok_{nome}"] += g is None
                    continue
                ok = (g is not None and g.num_nodes() == len(com_texto)
                      and [(ordem[s], ordem[t]) for s, t in zip(*[x.tolist() for x in g.edges()])] == arestas
                      and torch.equal(g.ndata["features"][:, 0], torch.arange(len(ordem), dtype=torch.float)))
                c[f"ok_{nome}"] += ok
            c["t5"] += TC.mesmo_grafo(ga, gn)
            if not descartar and go is not None:
                c["orig_carrega"] += 1
                c["orig_certo"] += [(ordem[s], ordem[t]) for s, t in zip(*[x.tolist() for x in go.edges()])] == arestas
            if descartar and go is not None:
                c["orig_carrega_com_no_sem_texto"] += 1
    print(f"  grafos {c['grafos']} (com nó sem texto nas arestas: {c['descartar_esperado']}); "
          f"original carrega {c['orig_carrega']} dos válidos e acerta {c['orig_certo']}; "
          f"original carrega {c['orig_carrega_com_no_sem_texto']} com nó sem texto")
    verificar(c["ok_antigo"] == n, f".pt antigo + loader corrigido: {c['ok_antigo']}/{n} exatamente certos "
                                   f"(arestas, ordem, direção, repetidas, laços, isolados) ou descartados quando devem")
    verificar(c["ok_novo"] == n, f".pt da extração corrigida + loader corrigido: {c['ok_novo']}/{n}")
    verificar(c["t5"] == n, f".pt antigo e novo dão o mesmo grafo: {c['t5']}/{n}")
    verificar(c["orig_certo"] < c["orig_carrega"], "sanidade: o código original erra nestes grafos (o fuzzing tem poder)")


# --------------------------------------------------------------------------- A3
ADVERSARIAL = r'''
#include <stdio.h>
/* "111111111111" -> "222222222222" num comentário */
struct s { char *m; };
int f(int a, struct s *p)
{
    char *x = "\"30064771072\" -> \"30064771073\"";
    char *y = "\"7\"->\"8\" [label = \"z\"]";
    char *w = "<BR/> & < > ' \\ \" ";
    // "999" -> "888"
    p->m = x;                       /* -> "5" */
    if (a > 0 && p->m[0] == '"')
        printf("%s %s %s \"1\" -> \"2\"\n", x, y, w);
    return a > 0 ? a : -a;
}
int g(void) { struct s q; return f(1, &q) - f(2, &q); }
'''


def arestas_regex_constanca(dot):
    """A leitura da Constança (parse_dot_edges): re.findall sobre o ficheiro inteiro."""
    return re.findall(r'"(\d+)"\s*->\s*"(\d+)"', Path(dot).read_text(encoding="utf-8", errors="ignore"))


def arestas_pydot(dot):
    import pydot
    (g,) = pydot.graph_from_dot_file(str(dot), encoding="utf-8")
    return [(e.get_source().strip('"'), e.get_destination().strip('"')) for e in g.get_edges()]


def a3():
    secao("A3  Arestas falsas: Joern com strings/comentários '\"N\" -> \"M\"'; 3 leitores do .dot")
    with tempfile.TemporaryDirectory() as tmp:
        c_file = Path(tmp) / "adv.c"
        c_file.write_text(ADVERSARIAL, encoding="utf-8")
        saidas = correr_joern(c_file, TC.TIPOS)
        dots = [p for t in TC.TIPOS for p in sorted(saidas[t].glob("*.dot"))]
        adv_dots = list(dots)
        dots += sorted(PIPELINE.glob("output/linux/*/*/*.dot"))
        iguais, total, n_arestas, aspas_cruas = 0, 0, 0, 0
        for d in dots:
            a = arestas_regex_constanca(d)
            b = [(s, t) for s, t, _ in ler_dot(d)[2]]
            c = arestas_pydot(d)
            total += 1
            n_arestas += len(a)
            # regex e parser de linhas: mesma sequência; pydot: mesmo multiconjunto (reordena arestas repetidas)
            iguais += a == b and Counter(b) == Counter(c)
            if a != b or Counter(b) != Counter(c):
                print(f"  DIFERENTE: {d.name}: regex {len(a)} linhas {len(b)} pydot {len(c)}")
        texto_adv = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in adv_dots)
        aspas_cruas = len(re.findall(r'"\d+"\s*->\s*"\d+"', texto_adv)) - sum(len(arestas_regex_constanca(p)) for p in adv_dots)
        apareceu = "30064771073\\&quot;" in texto_adv or "&quot;30064771072&quot;" in texto_adv
    print(f"  {len(adv_dots)} .dot do código adversarial + {total - len(adv_dots)} do 00_preparacao; {n_arestas} arestas")
    verificar(apareceu, "o texto adversarial chega ao .dot, com as aspas escapadas como &quot; (o teste exercita o caso)")
    verificar(iguais == total, f"regex da Constança == parser de linhas == pydot (multiconjunto) em {iguais}/{total} .dot")
    verificar(aspas_cruas == 0, "nenhum padrão '\"N\" -> \"M\"' no .dot para além das arestas verdadeiras")


# --------------------------------------------------------------------------- A4
def a4():
    secao("A4  Vetores reais do GraphCodeBERT: nó de cada linha identificado pelo vetor; arestas como pares de textos")
    textos_p, vet_p = GERADO / "impacto_tese" / "textos.pkl", GERADO / "impacto_tese" / "vetores.npy"
    if not vet_p.exists():
        verificar(False, "faltam textos.pkl/vetores.npy (testes_erro/impacto_tese/gerar_embeddings.py)")
        return
    with open(textos_p, "rb") as f:
        indice = {t: i for i, t in enumerate(pickle.load(f))}
    vet = np.load(vet_p, mmap_mode="r")
    c = Counter()
    for r in registos_amostra():
        textos = dict(r["nos"])
        emb = {n: torch.from_numpy(np.array(vet[indice[t]])) for n, t in textos.items()}
        g = TC.g_corrigido(pt_de_registo(r, emb))
        go = TC.g_original(pt_de_registo(r, emb))
        c["registos"] += 1
        if g is None:
            c["descartados"] += 1
            continue
        por_vetor = {}
        for t in set(textos.values()):
            por_vetor.setdefault(np.array(vet[indice[t]]).tobytes(), set()).add(t)
        linha_txt = []
        for v in g.ndata["features"].numpy():
            cand = por_vetor.get(v.astype(np.float32).tobytes(), set())
            linha_txt.append(next(iter(cand)) if len(cand) == 1 else None)
        c["linhas_ambiguas"] += sum(x is None for x in linha_txt)
        s, t = (x.tolist() for x in g.edges())
        vistas = Counter((linha_txt[a], linha_txt[b]) for a, b in zip(s, t))
        reais = Counter((textos[a], textos[b]) for a, b in r["arestas"])
        c["arestas"] += sum(reais.values())
        c["certas"] += sum((vistas & reais).values())
        c["grafos_certos"] += vistas == reais
        c["carregados"] += 1
        if go is not None:
            so, to = (x.tolist() for x in go.edges())
            c["orig_arestas"] += len(so)
            c["orig_certas"] += sum((Counter((linha_txt[a], linha_txt[b]) for a, b in zip(so, to)) & reais).values())
    print(f"  {c['registos']} grafos, {c['carregados']} carregados, {c['descartados']} descartados; "
          f"linhas sem vetor único: {c['linhas_ambiguas']}")
    print(f"  pares de textos certos: corrigido {c['certas']}/{c['arestas']}, original {c['orig_certas']}/{c['orig_arestas']}")
    verificar(c["linhas_ambiguas"] == 0, "cada linha da matriz é identificada pelo seu vetor real")
    verificar(c["certas"] == c["arestas"] and c["grafos_certos"] == c["carregados"],
              f"com features reais, {c['certas']}/{c['arestas']} arestas ligam os textos certos; "
              f"{c['grafos_certos']}/{c['carregados']} grafos exatos")
    verificar(c["orig_certas"] < 0.5 * c["orig_arestas"], "sanidade: o original falha a mesma verificação")


# --------------------------------------------------------------------------- A5
def a5():
    secao("A5  Caminhos do código que contornam a correção")
    repo = CLASSIFICACAO.parent
    # quem lê o edge_index GRAVADO no .pt (d['edge_index'] ou d.get('edge_index')) tem de aplicar a correção e estar
    # testado funcionalmente; quem só o constrói a partir do .dot não depende da numeração gravada
    usa_gravado = re.compile(r"""\[\s*['"]edge_index['"]\s*\]|\.get\(\s*['"]edge_index['"]""")
    testados = {"classification/embedding_dataset.py"}   # T7 de testar_correcao.py e A1 acima
    leitores, corrigidos, reconstroem, sem_correcao = [], [], [], []
    for p in sorted(repo.rglob("*.py")):
        txt = p.read_text(encoding="utf-8", errors="ignore")
        # create_embeddings.py escreve os .pt (e só relê os seus ao atualizar arestas): não é um leitor para o treino
        if "edge_index" in txt and "torch.load" in txt and p.name not in ("load_graphcodebert_embeddings.py",
                                                                          "create_embeddings.py"):
            rel = p.relative_to(repo).as_posix()
            leitores.append(rel)
            if not usa_gravado.search(txt):
                reconstroem.append(rel)
            elif "aligned_edge_index(" in txt:
                corrigidos.append(rel)
            else:
                sem_correcao.append(rel)
    print(f"  outros ficheiros que leem .pt: {leitores or 'nenhum'}")
    print(f"    usam o edge_index gravado, com a correção: {corrigidos or 'nenhum'}")
    print(f"    não usam o edge_index gravado (reconstroem as arestas a partir do .dot): {reconstroem or 'nenhum'}")
    print(f"    usam o edge_index gravado sem a correção: {sem_correcao or 'nenhum'}")
    verificar(not sem_correcao, "nenhum ficheiro usa o edge_index gravado sem a correção")
    verificar(set(corrigidos) <= testados, f"os que aplicam a correção estão testados (por testar: "
                                           f"{sorted(set(corrigidos) - testados) or 'nenhum'})")
    usados = [p for p in leitores if any(p.split('/')[-1][:-3] in q.read_text(encoding='utf-8', errors='ignore')
                                         for q in repo.rglob('*.py') if q.as_posix() != (repo / p).as_posix())]
    verificar(not usados, f"nenhum é importado pelo treino (importados: {usados or 'nenhum'})")
    if "extraction/fusion_all_graphs.py" in reconstroem:
        a5_fusion()
    # O SVM/XGBoost lê as embeddings de uma run fixa (DATA_DIR, uma run da tese). Corre-se o script verdadeiro numa
    # pasta vazia: no modo corrigido tem de recusar essa run; no original tem de passar a guarda (e falhar só por não
    # haver ficheiros de embeddings nesta máquina).
    import os
    import subprocess
    svm = CLASSIFICACAO / "new_classifiers_svm_xgboost.py"
    data_dir = re.search(r'^DATA_DIR\s*=\s*"([^"]+)"', svm.read_text(encoding="utf-8"), re.M).group(1)
    print(f"  new_classifiers_svm_xgboost.py: DATA_DIR = {data_dir}")
    saidas = {}
    for legacy in ("0", "1"):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run([sys.executable, "-X", "utf8", str(svm)], cwd=tmp, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300,
                               env=dict(os.environ, LEGACY_EDGE_ALIGNMENT=legacy, MPLBACKEND="Agg"))
        saidas[legacy] = (r.returncode, r.stdout + r.stderr)
    rc0, out0 = saidas["0"]
    rc1, out1 = saidas["1"]
    verificar(rc0 != 0 and "[fix 01] DATA_DIR does not match" in out0,
              "modo corrigido: o SVM/XGBoost recusa a run da tese (desalinhada) em vez de a ler em silêncio")
    verificar(rc1 != 0 and "[fix 01]" not in out1 and "dgcnn_embeddings_train_after.pt" in out1,
              "modo original: passa a guarda e só falha por faltarem as embeddings da tese nesta máquina")


def a5_fusion():
    """fusion_all_graphs.py não usa o edge_index gravado: constrói x e as arestas a partir do .dot, com a mesma ordem
    de nós. Confirma-se nos 18 .pt reais do ficheiro de teste, comparando cada aresta como par de VETORES com o .dot
    (um nó sem texto fica com o vetor nulo, como no fusion_all_graphs.py)."""
    from comum import EXTRACAO, carregar_definicoes
    pasta = PIPELINE / "dataset_final_embeddings"
    if not pasta.exists():
        verificar(False, "falta correr 00_preparacao/preparar.py")
        return
    fusao = carregar_definicoes(EXTRACAO / "fusion_all_graphs.py", {"process_fusion"},
                                {"DOTS_ROOT_FOLDER": str(PIPELINE)})["process_fusion"]
    def chave(v):
        return v.float().numpy().tobytes()
    certos = n = orig_errados = 0
    with tempfile.TemporaryDirectory() as tmp:
        for tipo in TC.TIPOS:
            for pt in sorted((pasta / tipo).glob("*.pt")):
                d = torch.load(pt, weights_only=False)
                saida = Path(tmp) / f"{tipo}_{pt.name}"
                if not fusao(str(pt), str(saida)):     # grafos com menos de 2 nós ou sem .dot
                    continue
                data = torch.load(saida, weights_only=False)
                _, nos, arestas_dot = ler_dot(PIPELINE / d["dot_path"])
                nulo = torch.zeros(data.x.shape[1])
                vetor = lambda nid: chave(d["embeddings"].get(nid, nulo))
                reais = Counter((vetor(a), vetor(b)) for a, b, _ in arestas_dot if a in nos and b in nos)
                s, t = data.edge_index.tolist()
                n += 1
                certos += Counter((chave(data.x[a]), chave(data.x[b])) for a, b in zip(s, t)) == reais
                go = TC.g_original(d)
                if go is not None:
                    f = go.ndata["features"]
                    so, to = (x.tolist() for x in go.edges())
                    orig_errados += Counter((chave(f[a]), chave(f[b])) for a, b in zip(so, to)) != reais
    verificar(n > 0 and certos == n, f"fusion_all_graphs.py: em {certos}/{n} grafos reais as arestas, como pares de "
                                     f"vetores, são as do .dot (alinhado por construção, não precisa da correção)")
    verificar(orig_errados > 0, f"sanidade: a mesma comparação falha em {orig_errados} grafos com o loader original "
                                f"(o teste tem poder)")


def main():
    cabecalho("01 - Testes adicionais da correção (mutação, fuzzing, Joern adversarial, vetores reais, bypass)")
    a1()
    a2()
    a3()
    a4()
    a5()
    print()
    if falhas:
        print(f"RESULTADO: {len(falhas)} VERIFICAÇÃO(ÕES) FALHARAM")
        for f in falhas:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: TODAS AS VERIFICAÇÕES PASSARAM")


if __name__ == "__main__":
    main()
