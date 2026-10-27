"""Caminhos e funções partilhadas pelos testes de verificação.

Os testes leem o repositório da Constança mas nunca o alteram. Tudo o que é gerado
vai para `_gerado/` (pode ser apagado a qualquer momento; o `00_preparacao` recria-o).
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# A consola do Windows usa cp1252 por omissão; os testes imprimem acentos e símbolos.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

RAIZ = Path(__file__).resolve().parent
TESE = RAIZ.parent
REPO = TESE / "uc.-msc-constanca-graphcodebert-vulnerability-detection"
EXTRACAO = REPO / "extraction"
CLASSIFICACAO = REPO / "classification"
DADOS = RAIZ / "dados"
GERADO = RAIZ / "_gerado"
PIPELINE = GERADO / "pipeline"          # cópia de trabalho da pipeline original sobre dados/vuln.c
JOERN = Path.home() / "joern-cli"
COMMIT_TESTE = "deadbeef" * 5
FICHEIRO_TESTE = "drivers/test/vuln.c"  # caminho "fingido" do ficheiro de teste dentro do repositório
TIPOS = ["ast", "cfg", "pdg"]


# --------------------------------------------------------------------------- saída
def cabecalho(titulo):
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def secao(titulo):
    print()
    print(f"--- {titulo} " + "-" * max(0, 72 - len(titulo)))


def veredicto(problema, gravidade, resumo):
    """problema: True (confirmado), False (sem problema) ou None (risco não confirmável)."""
    estado = {True: "PROBLEMA CONFIRMADO", False: "SEM PROBLEMA",
              None: "RISCO (comportamento confirmado, ocorrência não confirmável)"}[problema]
    print()
    print(f"VEREDICTO: {estado} | gravidade: {gravidade}")
    print(f"RESUMO: {resumo}")


def exigir_preparacao():
    if not (PIPELINE / "dataset_final_embeddings").exists():
        print("Falta correr 00_preparacao/preparar.py (ou correr_todos.py).")
        sys.exit(2)


# --------------------------------------------------------------------------- Joern
def _jar_c2cpg():
    jars = sorted((JOERN / "frontends" / "c2cpg" / "lib").glob("io.joern.c2cpg-*-launcher.jar"))
    if not jars:
        raise FileNotFoundError(f"c2cpg não encontrado em {JOERN}")
    return jars[-1]


def joern_parse(ficheiro_c, cpg_destino, extra_args=()):
    """Equivalente a `joern-parse` (o .bat falha em Windows; chama-se o frontend de C diretamente)."""
    r = subprocess.run(["java", "-jar", str(_jar_c2cpg()), str(ficheiro_c), "--output", str(cpg_destino), *extra_args],
                       capture_output=True, text=True)
    if not Path(cpg_destino).exists():
        raise RuntimeError(f"c2cpg falhou para {ficheiro_c}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def joern_export(cpg, repr_, destino):
    """Corre `joern-export --repr <repr_> --out <destino>`; devolve (returncode, saída)."""
    r = subprocess.run(["cmd", "/c", str(JOERN / "joern-export.bat"), str(cpg), "--repr", repr_, "--out", str(destino)],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def correr_joern(ficheiro_c, reprs, extra_args=()):
    """Parse + export num diretório temporário sem acentos nem espaços. Devolve {repr: Path}."""
    tmp = Path(tempfile.mkdtemp(prefix="verif_joern_"))
    src = tmp / Path(ficheiro_c).name
    shutil.copy(ficheiro_c, src)
    cpg = tmp / "cpg.bin"
    joern_parse(src, cpg, extra_args)
    saidas = {}
    for r in reprs:
        rc, out = joern_export(cpg, r, tmp / r)
        if rc != 0:
            raise RuntimeError(f"joern-export {r} falhou:\n{out[-2000:]}")
        saidas[r] = tmp / r
    return saidas


# --------------------------------------------------------------------------- .dot do Joern
RE_NO = re.compile(r'^"(\d+)" \[label = <(.*)> \]\s*$')
RE_ARESTA = re.compile(r'^\s*"(\d+)"\s*->\s*"(\d+)"\s*(?:\[\s*label\s*=\s*"(.*)"\s*\])?')


def ler_dot(caminho):
    """Parser independente do pydot. Devolve (nome_do_grafo, {id: label_html}, [(src, dst, label_aresta)])."""
    nos, arestas, nome = {}, [], None
    with open(caminho, encoding="utf-8", errors="replace") as f:
        for i, linha in enumerate(f):
            if i == 0:
                m = re.match(r'digraph "(.*)" \{', linha)
                nome = m.group(1) if m else None
                continue
            m = RE_NO.match(linha.strip())
            if m:
                nos[m.group(1)] = m.group(2)
                continue
            m = RE_ARESTA.match(linha)
            if m:
                arestas.append((m.group(1), m.group(2), m.group(3) or ""))
    return nome, nos, arestas


def campos_label(label_html):
    """'IDENTIFIER, 8<BR/>buf<BR/>buf[BUF_SIZE]' -> ['IDENTIFIER, 8', 'buf', 'buf[BUF_SIZE]'] (HTML já decodificado)."""
    import html
    return [html.unescape(p) for p in label_html.split("<BR/>")]


def nomes_dos_dots(pasta):
    """{ficheiro.dot: nome_da_função} numa pasta exportada pelo Joern."""
    import html
    return {p.name: html.unescape(ler_dot(p)[0] or "") for p in sorted(Path(pasta).glob("*.dot"),
                                                                      key=lambda p: int(p.name.split("-")[0]))}


def pasta_pipeline(tipo):
    return PIPELINE / "output" / "linux" / f"output-{tipo}-{COMMIT_TESTE}" / FICHEIRO_TESTE.replace("/", "---")


# --------------------------------------------------------------------------- dados reais do Linux
def ler_mapeamento(tipo, filtrado=True):
    """CSV de mapeamento função -> .dot produzido pela extração (output-data/)."""
    import pandas as pd
    nome = f"functions-{tipo}-linux{'-filtered' if filtrado else ''}.csv"
    return pd.read_csv(EXTRACAO / "output-data" / nome, sep=";")


def amostras_do_modelo(tipo):
    """Reproduz o que chega ao modelo: create_embeddings.py ignora linhas sem caminho e,
    para caminhos repetidos, fica com a PRIMEIRA linha (as seguintes são 'skipped')."""
    m = ler_mapeamento(tipo)
    m = m[m.CFG_filepath.notna()].drop_duplicates("CFG_filepath", keep="first").copy()
    m["y"] = (m.vulnerable_label.astype(str).str.lower() == "true").astype(int)
    return m


def ler_funcoes_fonte():
    """Todas as funções listadas em function-data/linux-functions.csv (dataset de origem)."""
    import pandas as pd
    src = pd.read_csv(EXTRACAO / "function-data" / "linux-functions.csv")
    src = src[src["Vulnerable File Functions"].notnull()]
    linhas = []
    for _, r in src.iterrows():
        for f in json.loads(r["Vulnerable File Functions"]):
            linhas.append((r["Vulnerable Commit Hash"], r["Vulnerable Author Date"], r["File Path"], f["Name"],
                           f["Signature"], f["Lines"][0], f["Lines"][1], f["Vulnerable"] == "Yes"))
    fn = pd.DataFrame(linhas, columns=["commit", "date", "file", "name", "sig", "l0", "l1", "vuln"])
    fn["nlines"] = fn.l1 - fn.l0
    return fn


# --------------------------------------------------------------------------- código da Constança
def ler_ast(ficheiro):
    return ast.parse(Path(ficheiro).read_text(encoding="utf-8"))


def carregar_definicoes(ficheiro, nomes, extra_globals=None):
    """Executa só os imports e as classes/funções pedidas de um script que corre tudo ao ser importado
    (caso de detect_vulnerabilities.py). O código das definições é o original, sem alterações."""
    ficheiro = Path(ficheiro)
    arvore = ler_ast(ficheiro)
    corpo = [n for n in arvore.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    corpo += [n for n in arvore.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in nomes]
    modulo = ast.Module(body=corpo, type_ignores=[])
    g = {"__name__": "verificacao", "__file__": str(ficheiro)}
    g.update(extra_globals or {})
    sys.path.insert(0, str(ficheiro.parent))
    cwd = os.getcwd()
    os.chdir(ficheiro.parent)
    try:
        exec(compile(modulo, str(ficheiro), "exec"), g)
    finally:
        os.chdir(cwd)
    return g


def atribuicoes_de_topo(ficheiro):
    """Valores literais atribuídos ao nível do módulo (configuração dos scripts)."""
    valores = {}
    for n in ler_ast(ficheiro).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                valores[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return valores
