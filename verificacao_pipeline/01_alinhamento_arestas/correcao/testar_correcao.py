"""01 - Testes da correção do alinhamento entre features e arestas.

Compara o código CORRIGIDO do repositório (loader e parse_dot_edges) com o ORIGINAL (comum.COMMIT_ORIGINAL):
  T0  o código da Constança está intacto (só linhas acrescentadas) e LEGACY_EDGE_ALIGNMENT=1 reproduz-o
  T1  casos sintéticos, sem Joern: o loader corrigido alinha as arestas; os casos-limite ficam como antes
  T2  os 18 grafos .pt do ficheiro de teste (Joern + extração original): 100% das arestas certas, mesmas labels
  T3  5984 grafos reais do kernel: 100% das arestas certas
  T4  nos mesmos grafos: a correção só muda as arestas (mesmos grafos carregados, mesmas features, mesma ordem)
  T5  .pt antigo + loader corrigido == .pt da extração corrigida + loader corrigido
  T6  loader corrigido == condição "corrigido" do testes_erro/impacto_tese/treinar.py (a que deu os números de DETALHES.md, secção 3)

Uso (a partir de verificacao_pipeline):
  python -X utf8 01_alinhamento_arestas/correcao/testar_correcao.py
T2 precisa do 00_preparacao; T3-T6 precisam da amostra de testes_erro/impacto_tese/construir_amostra.py. Demora ~2 min.
Sai com código 1 se algum teste falhar.
"""
import contextlib
import io
import os
import pickle
import subprocess
import sys
import tempfile
import types
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from comum import (CLASSIFICACAO, COMMIT_ORIGINAL, EXTRACAO, GERADO, PIPELINE, TIPOS, cabecalho, carregar_definicoes,
                   ler_dot, modulo_original, original, secao)

import dgl
import torch

# create_embeddings.py importa o transformers ao nível do módulo; só precisamos de parse_dot_edges.
_stub = types.ModuleType("transformers")
_stub.RobertaTokenizer = _stub.RobertaModel = None
sys.modules.setdefault("transformers", _stub)

ORIGINAL_LOADER = modulo_original("classification/load_graphcodebert_embeddings.py")
sys.path.insert(0, str(CLASSIFICACAO))
import load_graphcodebert_embeddings as CORRIGIDO  # noqa: E402  (versão corrigida, a do repositório)

parse_original = carregar_definicoes(original("extraction/create_embeddings.py"), {"parse_dot_edges"})["parse_dot_edges"]
_repo_ce = carregar_definicoes(EXTRACAO / "create_embeddings.py", {"parse_dot_edges", "parse_dot_edges_aligned"})
parse_corrigido = _repo_ce["parse_dot_edges_aligned"]      # a correção
parse_repo_original = _repo_ce["parse_dot_edges"]          # a função da Constança, que continua no repositório

# Ficheiros do repositório da Constança tocados pela correção (só com linhas acrescentadas).
FICHEIROS_CORRIGIDOS = ["classification/load_graphcodebert_embeddings.py", "classification/detect_vulnerabilities.py",
                        "extraction/create_embeddings.py"]

GRAFOS = GERADO / "impacto_tese" / "grafos"
VAZIO = torch.empty((2, 0), dtype=torch.long)
falhas = []


# --------------------------------------------------------------------------- utilitários
def verificar(condicao, descricao):
    print(f"  [{'ok' if condicao else 'FALHOU'}] {descricao}")
    if not condicao:
        falhas.append(descricao)


def silencioso(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def g_original(d):
    return silencioso(ORIGINAL_LOADER.create_dgl_graph_from_embedding, d)


def no_modo(legacy, f, *a):
    """Corre f com o loader do repositório no modo original (legacy=True) ou corrigido (False).
    É o mesmo que LEGACY_EDGE_ALIGNMENT=1 / 0 no ambiente (o T1 confirma que a variável faz isto)."""
    antes = CORRIGIDO.LEGACY_EDGE_ALIGNMENT
    CORRIGIDO.LEGACY_EDGE_ALIGNMENT = legacy
    try:
        return silencioso(f, *a)
    finally:
        CORRIGIDO.LEGACY_EDGE_ALIGNMENT = antes


def g_corrigido(d):
    return no_modo(False, CORRIGIDO.create_dgl_graph_from_embedding, d)


def g_modo_original(d):
    return no_modo(True, CORRIGIDO.create_dgl_graph_from_embedding, d)


def mesmo_parse(a, b):
    return torch.equal(a[0].reshape(2, -1), b[0].reshape(2, -1)) and a[1] == b[1]


def emb_codificado(ids):
    """A feature de cada nó é a sua posição na ordem numérica: assim sabe-se que nó está em cada linha."""
    ordem = sorted(ids, key=int)
    return {n: torch.tensor([float(i)]) for i, n in enumerate(ordem)}, ordem


def arestas_por_id(g, ordem):
    """Arestas de um grafo com features codificadas (emb_codificado), como pares de ids do Joern."""
    linhas = [ordem[int(v)] for v in g.ndata["features"][:, 0].tolist()]
    src, dst = g.edges()
    return Counter((linhas[s], linhas[t]) for s, t in zip(src.tolist(), dst.tolist()))


def arestas_por_vetor(g, emb):
    """Igual a arestas_por_id, para features reais: identifica o nó de cada linha comparando vetores."""
    feats = g.ndata["features"]
    linhas = {}
    for nid, v in emb.items():
        i = (feats == v.float()).all(dim=1).nonzero().flatten().tolist()
        assert len(i) == 1, "vetor não encontrado ou repetido"
        linhas[i[0]] = nid
    src, dst = g.edges()
    return Counter((linhas[s], linhas[t]) for s, t in zip(src.tolist(), dst.tolist()))


def mesmo_grafo(a, b):
    """Igualdade exata: mesmos nós pela mesma ordem, mesmas arestas pela mesma ordem, mesmas features."""
    if a is None or b is None:
        return a is b
    return (a.num_nodes() == b.num_nodes() and all(torch.equal(x, y) for x, y in zip(a.edges(), b.edges()))
            and torch.equal(a.ndata["features"], b.ndata["features"]))


def escrever_dot(caminho, arestas):
    """Um .dot mínimo: parse_dot_edges só lê as linhas de arestas."""
    linhas = ['digraph "f" {'] + [f'  "{a}" -> "{b}"  ' for a, b in arestas] + ["}"]
    Path(caminho).write_text("\n".join(linhas) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- T0
INTERRUPTOR = Path(__file__).resolve().parent / "interruptor.py"


def valor_do_interruptor():
    import ast
    for n in ast.parse(INTERRUPTOR.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "USAR_CODIGO_ORIGINAL":
            return ast.literal_eval(n.value)


def modo_em_processo_novo(ficheiro, env_valor):
    """Num processo Python novo, executa SÓ as linhas do interruptor do ficheiro verdadeiro do repositório
    (import os, EDGE_ALIGNMENT_SWITCH, read_edge_alignment_switch e a atribuição do modo), com __file__ no sítio
    real. Não corre o resto do script: o create_embeddings.py, importado inteiro, correria a extração."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("LEGACY_EDGE_ALIGNMENT", None)
    if env_valor is not None:
        env["LEGACY_EDGE_ALIGNMENT"] = env_valor
    codigo = """
import ast, sys
f = sys.argv[1]
arvore = ast.parse(open(f, encoding='utf-8').read())
nomes = {'EDGE_ALIGNMENT_SWITCH', 'LEGACY_EDGE_ALIGNMENT', 'EDGE_ALIGNMENT_FROM'}
manter = [n for n in arvore.body
          if (isinstance(n, ast.Import) and any(a.name == 'os' for a in n.names))
          or (isinstance(n, ast.FunctionDef) and n.name == 'read_edge_alignment_switch')
          or (isinstance(n, ast.Assign) and nomes & {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)})]
g = {'__file__': f, '__name__': 'verificacao'}
exec(compile(ast.Module(manter, []), f, 'exec'), g)
print('MODO', g['LEGACY_EDGE_ALIGNMENT'], g['EDGE_ALIGNMENT_FROM'])
"""
    r = subprocess.run([sys.executable, "-c", codigo, str(ficheiro)], env=env, capture_output=True, text=True, encoding="utf-8")
    linha = next((l for l in r.stdout.splitlines() if l.startswith("MODO ")), None)
    return linha.split(" ", 2)[1:] if linha else ["ERRO", r.stderr[-300:]]


def t0_interruptor():
    """O interruptor (correcao/interruptor.py) e a precedência da variável de ambiente."""
    print("  interruptor:", INTERRUPTOR, "| USAR_CODIGO_ORIGINAL =", valor_do_interruptor())
    leitor_ce = carregar_definicoes(EXTRACAO / "create_embeddings.py", {"read_edge_alignment_switch"},
                                    {"EDGE_ALIGNMENT_SWITCH": str(INTERRUPTOR)})["read_edge_alignment_switch"]
    verificar(Path(CORRIGIDO.EDGE_ALIGNMENT_SWITCH).resolve() == INTERRUPTOR.resolve(),
              "o loader procura o interruptor no sítio certo (esta pasta)")

    # 1) num processo novo, sem variável de ambiente: o loader e a extração obedecem ao ficheiro
    esperado = str(valor_do_interruptor())
    for modulo, ficheiro in (("load_graphcodebert_embeddings", CLASSIFICACAO / "load_graphcodebert_embeddings.py"),
                             ("create_embeddings", EXTRACAO / "create_embeddings.py")):
        valor, origem = modo_em_processo_novo(ficheiro, None)
        verificar(valor == esperado and Path(origem).resolve() == INTERRUPTOR.resolve(),
                  f"{modulo}: sem variável de ambiente segue o interruptor (USAR_CODIGO_ORIGINAL = {esperado})")
        valores = [modo_em_processo_novo(ficheiro, v)[0] for v in ("1", "0")]
        verificar(valores == ["True", "False"], f"{modulo}: LEGACY_EDGE_ALIGNMENT=1/0 ganha ao interruptor")

    # 2) a função de leitura, sobre ficheiros temporários (o interruptor verdadeiro nunca é alterado)
    antes = os.environ.pop("LEGACY_EDGE_ALIGNMENT", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            def ler(conteudo, leitor):
                f = Path(tmp) / "interruptor.py"
                if conteudo is None:
                    f.unlink(missing_ok=True)
                else:
                    f.write_text(conteudo, encoding="utf-8")
                try:
                    return leitor(str(f))[0]
                except (ValueError, SyntaxError) as e:
                    return type(e).__name__
            for nome, leitor in (("loader", CORRIGIDO.read_edge_alignment_switch), ("extração", leitor_ce)):
                casos = {"# comentário\nUSAR_CODIGO_ORIGINAL = True\n": True,
                         "USAR_CODIGO_ORIGINAL = False\n": False,
                         None: False,                                     # sem ficheiro: corrigido
                         "USAR_CODIGO_ORIGINAL = Tru\n": "ValueError",   # não é um literal
                         "USAR_CODIGO_ORIGINAL = 1\n": "ValueError",     # não é True/False
                         "USAR_CODIGO_ORIGINAL True\n": "SyntaxError",
                         "OUTRA = True\n": "ValueError"}                 # variável em falta
                obtidos = {k: ler(k, leitor) for k in casos}
                verificar(obtidos == casos, f"leitura do interruptor ({nome}): True, False, ficheiro em falta -> corrigido; "
                          f"ficheiro mal escrito -> erro (nunca escolhe um modo em silêncio)")
    finally:
        if antes is not None:
            os.environ["LEGACY_EDGE_ALIGNMENT"] = antes


def mesmos_campos(a, b, ignorar=()):
    """Mesmo .pt, exceto os VALORES dos vetores (o modelo-stub dá vetores aleatórios): compara os ids dos nós,
    o edge_index, o node_to_idx, a label e os restantes metadados."""
    if set(a) != set(b):
        return False
    for k in a:
        if k in ignorar:
            continue
        if k == "embeddings":
            if list(a[k]) != list(b[k]) or any(a[k][n].shape != b[k][n].shape for n in a[k]):
                return False
        elif isinstance(a[k], torch.Tensor):
            if not torch.equal(a[k], b[k]):
                return False
        elif a[k] != b[k]:
            return False
    return True


def correr_extracao_do_repositorio(destino, modo):
    """Corre o create_embeddings.py DO REPOSITÓRIO sobre uma cópia da pipeline de teste (como o 00_preparacao),
    com LEGACY_EDGE_ALIGNMENT=modo. Devolve (todos correram sem erro, linha 'Edge alignment' impressa)."""
    import re
    import shutil
    from comum import REPO
    shutil.copytree(PIPELINE, destino, ignore=shutil.ignore_patterns(
        "dataset_final_embeddings", "dataset_manifest.csv", "__pycache__", "create_embeddings_*.py"))
    fonte = (REPO / "extraction" / "create_embeddings.py").read_text(encoding="utf-8")
    ok, modos_impressos = True, set()
    for tipo in TIPOS:
        copia, n = re.subn(r'^GRAPH_TYPES = \["pdg"\]', f'GRAPH_TYPES = ["{tipo}"]', fonte, flags=re.M)
        assert n == 1
        script = destino / f"ce_{tipo}.py"
        script.write_text(copia, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", LEGACY_EDGE_ALIGNMENT=modo)
        r = subprocess.run([sys.executable, script.name, "--csv", f"output-data/functions-{tipo}-linux-filtered.csv"],
                           cwd=destino, capture_output=True, text=True, encoding="utf-8", env=env)
        ok &= r.returncode == 0
        modos_impressos |= {l.strip() for l in r.stdout.splitlines() if l.startswith("Edge alignment")}
    return ok, modos_impressos


def t0():
    secao("T0  O código da Constança está intacto, o modo original é o código dela e o interruptor funciona")
    import difflib
    from comum import REPO
    for rel in FICHEIROS_CORRIGIDOS:
        a = original(rel).read_text(encoding="utf-8").splitlines()
        b = (REPO / rel).read_text(encoding="utf-8").splitlines()
        ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
        mantidas = sum(i2 - i1 for op, i1, i2, _, _ in ops if op == "equal")
        acrescentadas = sum(j2 - j1 for op, _, _, j1, j2 in ops if op == "insert")
        verificar(all(op in ("equal", "insert") for op, *_ in ops) and mantidas == len(a),
                  f"{rel}: as {len(a)} linhas originais estão todas lá, pela mesma ordem; só há {acrescentadas} acrescentadas")

    t0_interruptor()

    if not (PIPELINE / "output-data").exists():
        verificar(False, "falta correr 00_preparacao/preparar.py")
        return
    referencia = {p.relative_to(PIPELINE): torch.load(p, weights_only=False)
                  for p in sorted((PIPELINE / "dataset_final_embeddings").rglob("*.pt"))}   # feitos com o código original
    with tempfile.TemporaryDirectory() as tmp:
        for modo in ("1", "0"):
            destino = Path(tmp) / f"modo_{modo}"
            ok, impressos = correr_extracao_do_repositorio(destino, modo)   # a variável de ambiente ganha ao interruptor
            novos = {p.relative_to(destino): torch.load(p, weights_only=False)
                     for p in sorted((destino / "dataset_final_embeddings").rglob("*.pt"))}
            mesmos_ficheiros = ok and set(novos) == set(referencia)
            if modo == "1":
                iguais = sum(mesmos_campos(novos[k], referencia[k]) for k in referencia if k in novos)
                verificar(mesmos_ficheiros and iguais == len(referencia) and impressos == {"Edge alignment: ORIGINAL (original code, no fix) [from environment variable LEGACY_EDGE_ALIGNMENT]"},
                          f"extração do repositório com LEGACY_EDGE_ALIGNMENT=1: {iguais}/{len(referencia)} .pt iguais aos "
                          f"do código original (edge_index, node_to_idx, ids, labels, metadados)")
            else:
                # .pt novo com os vetores do .pt de referência: o loader corrigido tem de dar o mesmo grafo
                iguais = sum(mesmos_campos(novos[k], referencia[k], ignorar={"edge_index", "node_to_idx"})
                             and mesmo_grafo(g_corrigido(dict(novos[k], embeddings=referencia[k]["embeddings"])),
                                             g_corrigido(referencia[k]))
                             for k in referencia if k in novos)
                verificar(mesmos_ficheiros and iguais == len(referencia) and impressos == {"Edge alignment: fixed [from environment variable LEGACY_EDGE_ALIGNMENT]"},
                          f"extração do repositório sem a variável (corrigida): {iguais}/{len(referencia)} .pt dão o mesmo "
                          f"grafo corrigido; o resto do .pt não muda")


# --------------------------------------------------------------------------- T1
def t1():
    secao("T1  Casos sintéticos (sem Joern)")
    # 12 dígitos (METHOD, METHOD_RETURN) e 11 dígitos (CALL, IDENTIFIER): as duas ordens divergem
    M, R, C1, C2, I, ISO = "107374182400", "128849018880", "30064771072", "30064771073", "68719476736", "68719476737"
    SEM_TEXTO = "30064771099"
    arestas = [(M, C1), (C1, C2), (C2, I), (I, R), (M, R), (C1, C2)]     # com uma aresta repetida; ISO fica isolado
    emb, ordem = emb_codificado([M, R, C1, C2, I, ISO])
    verdade = Counter(arestas)
    with tempfile.TemporaryDirectory() as tmp:
        dot, dot_p, dot_0 = Path(tmp) / "f.dot", Path(tmp) / "p.dot", Path(tmp) / "z.dot"
        escrever_dot(dot, arestas)
        escrever_dot(dot_p, arestas + [(C2, SEM_TEXTO)])
        escrever_dot(dot_0, [])
        ei, n2i = parse_original(str(dot))
        antigo = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
        ei, n2i = parse_corrigido(str(dot), emb.keys())
        novo = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
        ei, n2i = parse_original(str(dot_p))
        antigo_p = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
        ei, n2i = parse_corrigido(str(dot_p), emb.keys())
        novo_p = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}
        ei, n2i = parse_corrigido(str(dot_0), emb.keys())
        novo_0 = {"embeddings": emb, "edge_index": ei, "node_to_idx": n2i}

    go, gc, gn = g_original(antigo), g_corrigido(antigo), g_corrigido(novo)
    verificar(arestas_por_id(go, ordem) != verdade,
              "sanidade: com o código original as arestas ficam erradas (o teste deteta o erro)")
    verificar(gc is not None and arestas_por_id(gc, ordem) == verdade and gc.num_nodes() == len(emb),
              ".pt antigo + loader corrigido: todas as arestas certas, com o nó isolado")
    verificar(gc is not None and torch.equal(gc.ndata["features"], go.ndata["features"]),
              ".pt antigo + loader corrigido: matriz de features igual à do loader original (só mudam as arestas)")
    verificar(mesmo_grafo(gn, gc), ".pt da extração corrigida + loader corrigido: grafo idêntico ao anterior")
    verificar(arestas_por_id(g_original(novo), ordem) == verdade,
              ".pt da extração corrigida + loader original: também fica certo")
    verificar(mesmo_grafo(g_modo_original(antigo), go),
              "modo original (LEGACY_EDGE_ALIGNMENT=1) reproduz exatamente o grafo do código original")

    verificar(g_corrigido(antigo_p) is None and g_corrigido(novo_p) is None,
              "aresta para um id sem texto: o grafo é descartado (.pt antigo e novo)")
    verificar(g_corrigido({"embeddings": emb, "edge_index": antigo["edge_index"]}) is None,
              "edge_index sem node_to_idx: o grafo é descartado (não se pode alinhar)")
    estragado = dict(antigo, node_to_idx={n: i + 1 for n, i in antigo["node_to_idx"].items()})
    verificar(g_corrigido(estragado) is None, "node_to_idx que não é 0..n-1: o grafo é descartado")
    g0 = g_corrigido({"embeddings": emb, "edge_index": VAZIO, "node_to_idx": {}})
    verificar(g0 is not None and g0.num_edges() == 0 and g0.num_nodes() == len(emb),
              "grafo sem arestas: carrega com todos os nós, como no original")
    verificar(mesmo_grafo(g_corrigido(novo_0), g0), "grafo sem arestas da extração corrigida: igual")
    verificar(g_corrigido({"embeddings": {}, "edge_index": VAZIO, "node_to_idx": {}}) is None
              and g_original({"embeddings": {}, "edge_index": VAZIO, "node_to_idx": {}}) is None,
              "grafo sem nós: descartado, como no original")



# --------------------------------------------------------------------------- T2
def t2():
    secao("T2  Ficheiro de teste: 18 grafos .pt reais (Joern + extração original, dados/vuln.c)")
    pasta = PIPELINE / "dataset_final_embeddings"
    if not pasta.exists():
        verificar(False, "falta correr 00_preparacao/preparar.py")
        return
    total = certas_o = certas_c = grafos_certos = iguais_novo = n = modo_original_igual = parse_igual = 0
    for tipo in TIPOS:
        for pt in sorted((pasta / tipo).glob("*.pt")):
            d = torch.load(pt, weights_only=False)
            dot = PIPELINE / d["dot_path"]
            verdade = Counter((a, b) for a, b, _ in ler_dot(dot)[2])
            go, gc = g_original(d), g_corrigido(d)
            vistas_c = arestas_por_vetor(gc, d["embeddings"])
            certas_o += sum((arestas_por_vetor(go, d["embeddings"]) & verdade).values())
            certas_c += sum((vistas_c & verdade).values())
            grafos_certos += vistas_c == verdade
            total += sum(verdade.values())
            ei, n2i = parse_corrigido(str(dot), d["embeddings"].keys())        # T5 com o .dot real
            iguais_novo += mesmo_grafo(g_corrigido(dict(d, edge_index=ei, node_to_idx=n2i)), gc)
            modo_original_igual += mesmo_grafo(g_modo_original(d), go)                       # T0
            parse_igual += mesmo_parse(parse_repo_original(str(dot)), parse_original(str(dot)))  # T0
            n += 1
    print(f"  arestas certas: original {certas_o}/{total} ({certas_o / total * 100:.0f}%), corrigido {certas_c}/{total}")
    verificar(certas_c == total and grafos_certos == n, f"loader corrigido: {certas_c}/{total} arestas certas, "
              f"{grafos_certos}/{n} grafos exatamente iguais ao .dot")
    verificar(iguais_novo == n, f"T5 no .dot real: .pt antigo e .pt da extração corrigida dão o mesmo grafo em {iguais_novo}/{n}")
    verificar(modo_original_igual == n, f"T0: modo original do loader == loader original em {modo_original_igual}/{n}")
    verificar(parse_igual == n, f"T0: parse_dot_edges do repositório == a original em {parse_igual}/{n} .dot reais")

    iguais = iguais_legacy = True
    for tipo in TIPOS:
        do = silencioso(ORIGINAL_LOADER.load_embeddings_from_directory, str(pasta), tipo)
        dc = no_modo(False, CORRIGIDO.load_embeddings_from_directory, str(pasta), tipo)
        dl = no_modo(True, CORRIGIDO.load_embeddings_from_directory, str(pasta), tipo)
        iguais &= (list(do.label) == list(dc.label) and len(do) == len(dc)
                   and all(torch.equal(a.ndata["features"], b.ndata["features"]) for a, b in zip(do.graphs, dc.graphs)))
        iguais_legacy &= (list(do.label) == list(dl.label) and len(do) == len(dl)
                          and all(mesmo_grafo(a, b) for a, b in zip(do.graphs, dl.graphs)))
    verificar(iguais, "load_embeddings_from_directory: mesmos grafos, mesmas labels e features que o original")
    verificar(iguais_legacy, "T0: load_embeddings_from_directory no modo original == original (grafos, arestas e labels)")


# --------------------------------------------------------------------------- T3 a T6
def t3_a_t6():
    secao("T3-T6  Amostra real do kernel (testes_erro/impacto_tese/construir_amostra.py)")
    ficheiros = sorted(GRAFOS.glob("*.pkl"))
    if not ficheiros:
        verificar(False, "falta a amostra (testes_erro/impacto_tese/construir_amostra.py)")
        return
    s = defaultdict(Counter)
    exemplos = defaultdict(list)
    with tempfile.TemporaryDirectory() as tmp:
        dot = Path(tmp) / "g.dot"
        for p in ficheiros:
            with open(p, "rb") as f:
                registos = pickle.load(f)["grafos"]
            for r in registos:
                t = r["tipo"]
                textos = dict(r["nos"])
                emb, ordem = emb_codificado(textos)
                d = {"embeddings": emb, "edge_index": torch.tensor(r["edge_index"], dtype=torch.long).reshape(2, -1),
                     "node_to_idx": r["node_to_idx"]}
                go, gc = g_original(d), g_corrigido(d)
                s[t]["registos"] += 1
                s[t]["t0_loader"] += mesmo_grafo(g_modo_original(d), go)
                s[t]["carregados_original"] += go is not None
                s[t]["carregados_corrigido"] += gc is not None
                if go is not None and gc is None:
                    sem_texto = sum(1 for n in r["node_to_idx"] if n not in textos)
                    exemplos[t].append(f"{r['funcao']} ({len(textos)} nós, {sem_texto} id(s) nas arestas sem texto)")
                if gc is not None and go is None:
                    s[t]["so_corrigido"] += 1

                # T3: arestas do grafo corrigido == arestas do .dot (lidas pelo parser independente)
                if gc is not None:
                    vistas, reais = arestas_por_id(gc, ordem), Counter(map(tuple, r["arestas"]))
                    s[t]["arestas"] += sum(reais.values())
                    s[t]["arestas_certas"] += sum((vistas & reais).values())
                    s[t]["grafos_certos"] += vistas == reais
                # T4: só mudam as arestas
                if gc is not None and go is not None:
                    s[t]["ambos"] += 1
                    s[t]["t4_iguais"] += (gc.num_nodes() == go.num_nodes() and gc.num_edges() == go.num_edges()
                                          and torch.equal(gc.ndata["features"], go.ndata["features"]))
                # T5: refazer o edge_index com a extração corrigida, a partir das mesmas arestas
                escrever_dot(dot, r["arestas"])
                ei_o, n2i_o = parse_original(str(dot))
                s[t]["t0_parse"] += mesmo_parse(parse_repo_original(str(dot)), (ei_o, n2i_o))
                s[t]["t5_reconstrucao_ok"] += torch.equal(ei_o.reshape(2, -1), d["edge_index"]) and n2i_o == r["node_to_idx"]
                ei_c, n2i_c = parse_corrigido(str(dot), emb.keys())
                s[t]["t5_iguais"] += mesmo_grafo(g_corrigido({"embeddings": emb, "edge_index": ei_c, "node_to_idx": n2i_c}), gc)
                # T6: o treinar.py usa os grafos que o loader original carrega e refaz as arestas a partir do .dot
                if go is not None:
                    idx = {n: i for i, n in enumerate(ordem)}
                    pares = [(idx[a], idx[b]) for a, b in r["arestas"] if a in idx and b in idx]
                    gt = dgl.graph(([a for a, _ in pares], [b for _, b in pares]), num_nodes=len(ordem))
                    gt.ndata["features"] = go.ndata["features"]
                    s[t]["t6_treinar"] += 1
                    s[t]["t6_iguais"] += mesmo_grafo(gt, gc)

    print(f"  {'tipo':4} {'registos':>8} | {'carregados orig.':>16} {'corrig.':>7} | {'arestas certas (corrigido)':>27} "
          f"{'grafos 100%':>13} | {'T4':>11} | {'T5':>11} | {'T6':>11}")
    for t in TIPOS:
        x = s[t]
        print(f"  {t:4} {x['registos']:8} | {x['carregados_original']:16} {x['carregados_corrigido']:7} | "
              f"{x['arestas_certas']:>12}/{x['arestas']:<8} {x['arestas_certas'] / max(1, x['arestas']) * 100:5.1f}% "
              f"{x['grafos_certos']:>6}/{x['carregados_corrigido']:<6} | {x['t4_iguais']:>5}/{x['ambos']:<5} | "
              f"{x['t5_iguais']:>5}/{x['registos']:<5} | {x['t6_iguais']:>5}/{x['t6_treinar']:<5}")
        for e in exemplos[t]:
            print(f"       carregado pelo original e descartado pelo corrigido: {e}")
    tot = lambda k: sum(s[t][k] for t in TIPOS)

    print()
    verificar(tot("t0_loader") == tot("registos") and tot("t0_parse") == tot("registos"),
              f"T0: modo original == código original em {tot('t0_loader')}/{tot('registos')} grafos (loader) e "
              f"{tot('t0_parse')}/{tot('registos')} (parse_dot_edges)")
    verificar(tot("t5_reconstrucao_ok") == tot("registos"),
              f"(apoio) o .dot reconstruído reproduz o edge_index original em {tot('t5_reconstrucao_ok')}/{tot('registos')}")
    verificar(tot("arestas_certas") == tot("arestas") and tot("grafos_certos") == tot("carregados_corrigido"),
              f"T3: {tot('arestas_certas')}/{tot('arestas')} arestas certas; "
              f"{tot('grafos_certos')}/{tot('carregados_corrigido')} grafos exatamente iguais ao .dot")
    descartados = tot("carregados_original") - tot("ambos")
    verificar(tot("so_corrigido") == 0 and descartados == sum(len(v) for v in exemplos.values()) and descartados <= 2,
              f"T4: nenhum grafo novo; {descartados} grafo(s) passam a ser descartados (o original carregava-os "
              f"com uma aresta para um nó sem texto)")
    verificar(tot("t4_iguais") == tot("ambos"),
              f"T4: nos {tot('ambos')} grafos carregados pelos dois, mesmos nós, nº de arestas e features (mesma ordem)")
    verificar(tot("t5_iguais") == tot("registos"),
              f"T5: .pt antigo e .pt da extração corrigida dão o mesmo grafo em {tot('t5_iguais')}/{tot('registos')}")
    verificar(tot("t6_iguais") == tot("t6_treinar") - descartados,
              f"T6: grafo idêntico ao da condição 'corrigido' do treinar.py em {tot('t6_iguais')}/{tot('t6_treinar')} "
              f"(os restantes {descartados} são os descartados acima)")


def main():
    cabecalho("01 - Testes da correção do alinhamento (código corrigido vs original)")
    print(f"código original: commit {COMMIT_ORIGINAL[:7]} | corrigido: o do repositório")
    t0()
    t1()
    t2()
    t3_a_t6()
    print()
    if falhas:
        print(f"RESULTADO: {len(falhas)} TESTE(S) FALHARAM")
        for f in falhas:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: TODOS OS TESTES PASSARAM")


if __name__ == "__main__":
    main()
