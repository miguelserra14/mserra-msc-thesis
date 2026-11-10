"""Reconstrói uma amostra REAL do dataset da tese com o código original da extração.

Sorteia pares (commit, ficheiro) com pelo menos uma função vulnerável nos CSV reais. De cada um guarda todas as
vulneráveis e no máximo MAX_NEGATIVAS não vulneráveis. Descarrega o ficheiro do kernel nesse commit, corre o Joern
sem headers (como a pipeline), gera os .txt com generate_txt_force e o edge_index com parse_dot_edges.
Grava um .pkl por ficheiro em _gerado/impacto_tese/grafos/ (o trabalho é retomável).

Uso: python -X utf8 construir_amostra.py --n-ficheiros 160 --workers 2
"""
import argparse
import contextlib
import ctypes
import hashlib
import html
import io
import json
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import types
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
from comum import EXTRACAO, GERADO, JOERN, TIPOS, _jar_c2cpg, amostras_do_modelo, carregar_definicoes, ler_dot, original

SAIDA = GERADO / "impacto_tese"
SEED = 20261031
MAX_NEGATIVAS = 15


def pares_sorteados():
    """Mesma ordem em todas as execuções: amostras maiores contêm as mais pequenas."""
    linhas = {t: amostras_do_modelo(t) for t in TIPOS}
    comuns = None
    for m in linhas.values():
        p = set(zip(m.commit, m.filepath))
        comuns = p if comuns is None else comuns & p
    com_vuln = set(zip(*[linhas["pdg"].loc[linhas["pdg"].y == 1, c] for c in ("commit", "filepath")]))
    pares = sorted(comuns & com_vuln)
    random.Random(SEED).shuffle(pares)
    return pares, linhas


def escolher_negativas(linhas_por_tipo, commit, ficheiro):
    """A mesma seleção de funções nos 3 tipos: todas as vulneráveis e até MAX_NEGATIVAS das restantes."""
    vuln = {f for ls in linhas_por_tipo.values() for f, _, y in ls if y == 1}
    outras = sorted({f for ls in linhas_por_tipo.values() for f, _, _ in ls} - vuln)
    random.Random(f"{SEED}:{commit}:{ficheiro}").shuffle(outras)
    manter = vuln | set(outras[:MAX_NEGATIVAS])
    return {t: [l for l in ls if l[0] in manter] for t, ls in linhas_por_tipo.items()}


def chave(commit, ficheiro):
    return hashlib.sha1(f"{commit}:{ficheiro}".encode()).hexdigest()[:16]


def preparar_repo():
    """Repositório parcial: só traz os commits e os ficheiros pedidos."""
    repo = SAIDA / "repo_kernel"
    if not repo.exists():
        subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", "https://github.com/torvalds/linux.git"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "remote.origin.promisor", "true"], check=True)
        subprocess.run(["git", "-C", str(repo), "config", "remote.origin.partialclonefilter", "blob:none"], check=True)
    return repo


def descarregar(repo, commit, ficheiro, destino):
    for tentativa in range(3):
        try:
            subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", commit],
                           check=True, capture_output=True, timeout=300)
            r = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{ficheiro}"], check=True, capture_output=True, timeout=300)
            destino.write_bytes(r.stdout)
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            time.sleep(15 * (tentativa + 1))
    return False


def baixar_prioridade():
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)  # abaixo do normal


_definicoes = {}


def _carregar_originais():
    if _definicoes:
        return _definicoes
    stub = types.ModuleType("transformers")
    stub.RobertaTokenizer = stub.RobertaModel = None
    sys.modules.setdefault("transformers", stub)
    _definicoes["parse_dot_edges"] = carregar_definicoes(original("extraction/create_embeddings.py"), {"parse_dot_edges"})["parse_dot_edges"]
    sys.path.insert(0, str(EXTRACAO))
    from graph_extraction import generate_txt_force
    _definicoes["generate_txt_force"] = generate_txt_force
    return _definicoes


def ler_txt(caminho):
    """Igual ao passo B de create_embeddings.py."""
    nos = []
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        for linha in f:
            p = linha.strip().split(" ", 1)
            if len(p) == 2:
                nos.append((p[0], p[1]))
    return nos


def processar(commit, ficheiro, fonte, linhas_por_tipo):
    """linhas_por_tipo: {tipo: [(function_name, numero_do_dot, y)]}"""
    orig = _carregar_originais()
    inicio = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="impacto_"))
    r = {"commit": commit, "ficheiro": ficheiro, "estado": "ok", "grafos": [], "sem_dot": 0, "mapa_indice": 0, "mapa_nome": 0}
    try:
        src = tmp / Path(ficheiro).name
        shutil.copy(fonte, src)
        cpg = tmp / "cpg.bin"
        subprocess.run(["java", "-jar", str(_jar_c2cpg()), str(src), "--output", str(cpg)],
                       capture_output=True, text=True, timeout=1200)
        if not cpg.exists():
            r["estado"] = "parse_falhou"
            return r
        for tipo in TIPOS:
            pasta = tmp / tipo
            subprocess.run(["cmd", "/c", str(JOERN / "joern-export.bat"), str(cpg), "--repr", tipo, "--out", str(pasta)],
                           capture_output=True, text=True, timeout=1200)
            dots = {p.name: p for p in pasta.glob("*.dot")}
            nomes = {nome: html.unescape(ler_dot(p)[0] or "") for nome, p in dots.items()}
            por_nome = {}
            for nome in sorted(nomes, key=lambda n: int(n.split("-")[0])):
                por_nome.setdefault(nomes[nome], nome)
            for funcao, numero, y in linhas_por_tipo[tipo]:
                alvo = f"{numero}-{tipo}.dot"
                if nomes.get(alvo) == funcao:
                    r["mapa_indice"] += 1
                elif funcao in por_nome:
                    alvo = por_nome[funcao]
                    r["mapa_nome"] += 1
                else:
                    r["sem_dot"] += 1
                    continue
                dot = dots[alvo]
                with contextlib.redirect_stdout(io.StringIO()):
                    orig["generate_txt_force"](str(dot))
                txt = dot.with_name(dot.name + ".txt")
                if not txt.exists():
                    r["sem_dot"] += 1
                    continue
                nos = ler_txt(txt)
                edge_index, node_to_idx = orig["parse_dot_edges"](str(dot))
                r["grafos"].append({
                    "tipo": tipo, "funcao": funcao, "y": int(y), "nos": nos,
                    "edge_index": edge_index.tolist(), "node_to_idx": node_to_idx,
                    "arestas": [(a, b) for a, b, _ in ler_dot(dot)[2]],
                })
    except subprocess.TimeoutExpired:
        r["estado"] = "joern_timeout"
    except Exception as e:
        r["estado"] = f"erro: {type(e).__name__}: {e}"[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r["segundos"] = time.time() - inicio
    return r


def tarefa(args):
    commit, ficheiro, fonte, linhas, destino = args
    r = processar(commit, ficheiro, fonte, linhas)
    with open(destino, "wb") as f:
        pickle.dump(r, f, protocol=4)
    return {k: v for k, v in r.items() if k != "grafos"} | {"n_grafos": len(r["grafos"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ficheiros", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pasta = SAIDA / "grafos"
    pasta.mkdir(parents=True, exist_ok=True)
    fontes = SAIDA / "src"
    fontes.mkdir(exist_ok=True)
    pares, linhas = pares_sorteados()
    print(f"{len(pares)} pares (commit, ficheiro) comuns aos 3 tipos; a processar os primeiros {a.n_ficheiros}", flush=True)

    grupos = {t: {k: list(zip(g.function_name, g.CFG_filepath.str.extract(r"/(\d+)-[a-z]+\.dot$")[0], g.y))
                  for k, g in m.groupby(["commit", "filepath"])} for t, m in linhas.items()}
    repo = preparar_repo()
    t0 = time.time()
    pendentes = set()

    def recolher(log, bloquear):
        feitos, _ = wait(pendentes, return_when=FIRST_COMPLETED, timeout=None if bloquear else 0)
        for fut in feitos:
            pendentes.discard(fut)
            r = fut.result()
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()
            print(f"[{(time.time() - t0) / 60:.1f} min] {r['estado']} {r['ficheiro']} grafos={r['n_grafos']} "
                  f"sem_dot={r['sem_dot']} {r.get('segundos', 0):.0f}s", flush=True)

    with ProcessPoolExecutor(a.workers, initializer=baixar_prioridade) as ex, open(SAIDA / "registo.jsonl", "a", encoding="utf-8") as log:
        for commit, ficheiro in pares[:a.n_ficheiros]:
            destino = pasta / f"{chave(commit, ficheiro)}.pkl"
            fonte = fontes / f"{chave(commit, ficheiro)}{Path(ficheiro).suffix}"
            if destino.exists():
                continue
            if not fonte.exists() and not descarregar(repo, commit, ficheiro, fonte):
                print(f"sem download: {commit[:10]} {ficheiro}", flush=True)
                continue
            linhas_tarefa = escolher_negativas({t: grupos[t][(commit, ficheiro)] for t in TIPOS}, commit, ficheiro)
            while len(pendentes) >= a.workers + 1:          # não acumula demasiados ficheiros à espera do Joern
                recolher(log, True)
            pendentes.add(ex.submit(tarefa, (commit, ficheiro, fonte, linhas_tarefa, destino)))
            recolher(log, False) if pendentes else None
        while pendentes:
            recolher(log, True)
    print("FIM", flush=True)


if __name__ == "__main__":
    main()
