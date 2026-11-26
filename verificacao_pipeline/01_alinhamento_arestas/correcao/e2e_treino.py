"""01 - Teste de ponta a ponta: o treino da Constança, sem alterações, com a correção ligada.

1. Escreve .pt no formato exato do create_embeddings.py (vetores reais do GraphCodeBERT, edge_index e node_to_idx
   da extração ORIGINAL, isto é, .pt como os da tese) para uma amostra de grafos PDG reais.
2. Planta em datasets_embeddings/ uma cache antiga (pdg-graphcodebert-embeddings.pkl, feita com o loader original),
   como estará na máquina da tese.
3. Corre classification/detect_vulnerabilities.py TAL COMO ESTÁ (T12, T10, T9, T11; 5 runs cada), com EMBEDDING_DIR.
4. Verifica: a cache antiga não foi usada nem alterada; a cache nova (-aligned) tem todas as arestas certas
   (identificando cada nó pelo vetor real); as runs ficam em pastas '-aligned' com as embeddings gravadas;
   e as 4 configurações chegaram ao fim.

Com --runs 1 corre-se uma cópia do detect_vulnerabilities.py em que só muda N_RUNS (verificado): as 5 runs só
repetem o treino com outras seeds e não exercitam mais código; com 1 run o teste demora ~3 h em vez de ~12 h.

Uso (a partir de verificacao_pipeline):
  python -X utf8 01_alinhamento_arestas/correcao/e2e_treino.py [--n 600] [--runs 5] [--so-verificar]
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import testar_correcao as TC  # noqa: E402
from comum import CLASSIFICACAO, GERADO, cabecalho, secao  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

TRAB = GERADO / "correcao_e2e"
EMB = TRAB / "emb"
CWD = TRAB / "cwd"
verificar, falhas = TC.verificar, TC.falhas


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def escolher(n, seed=0):
    regs = [r for p in sorted(TC.GRAFOS.glob("*.pkl")) for r in pickle.load(open(p, "rb"))["grafos"] if r["tipo"] == "pdg"]
    rng = random.Random(seed)
    pos = [r for r in regs if r["y"] == 1]
    neg = [r for r in regs if r["y"] == 0]
    rng.shuffle(pos), rng.shuffle(neg)
    k = min(len(pos), n // 4)
    return pos[:k] + neg[:n - k]


def preparar(n):
    secao(f"1-2. {n} .pt PDG no formato do create_embeddings.py + cache antiga plantada")
    with open(GERADO / "impacto_tese" / "textos.pkl", "rb") as f:
        indice = {t: i for i, t in enumerate(pickle.load(f))}
    vet = np.load(GERADO / "impacto_tese" / "vetores.npy", mmap_mode="r")
    (EMB / "pdg").mkdir(parents=True, exist_ok=True)
    (CWD / "datasets_embeddings").mkdir(parents=True, exist_ok=True)
    verdade = {}
    for i, r in enumerate(escolher(n)):
        nome = f"{i:04d}_{r['funcao']}.pt"
        emb = {nid: torch.from_numpy(np.array(vet[indice[t]])) for nid, t in r["nos"]}
        ei = torch.tensor(r["edge_index"], dtype=torch.long).reshape(2, -1)
        torch.save({"embeddings": emb, "label": torch.tensor(r["y"], dtype=torch.long), "function_name": r["funcao"],
                    "dot_path": f"amostra/{r['funcao']}-pdg.dot", "graph_type": "pdg", "commit": "amostra",
                    "filepath": "amostra", "num_nodes": len(emb), "pooling_strategy": "mean",
                    "edge_index": ei, "node_to_idx": r["node_to_idx"], "num_edges": ei.shape[1]}, EMB / "pdg" / nome)
        verdade[nome] = {"y": r["y"], "textos": dict(r["nos"]), "arestas": [tuple(a) for a in r["arestas"]]}
    with open(TRAB / "verdade.pkl", "wb") as f:
        pickle.dump(verdade, f)
    # cache antiga, feita com o loader ORIGINAL (o que existe na máquina da tese)
    df = TC.silencioso(TC.ORIGINAL_LOADER.load_embeddings_from_directory, str(EMB), "pdg")
    antiga = CWD / "datasets_embeddings" / "pdg-graphcodebert-embeddings.pkl"
    df.to_pickle(antiga)
    (TRAB / "cache_antiga.sha").write_text(sha(antiga))
    print(f"  {len(verdade)} .pt ({sum(v['y'] for v in verdade.values())} vulneráveis); cache antiga com {len(df)} grafos")


def script_de_treino(runs):
    """O detect_vulnerabilities.py do repositório. Com runs != 5, uma cópia em TRAB em que só muda a linha
    'N_RUNS = 5' (as seeds saem de N_RUNS: a run 1 tem sempre a seed 42). Verifica que é a única diferença."""
    original = CLASSIFICACAO / "detect_vulnerabilities.py"
    if runs == 5:
        return original
    import difflib
    import re
    fonte = original.read_text(encoding="utf-8")
    copia, n = re.subn(r"^N_RUNS = 5\b", f"N_RUNS = {runs}", fonte, flags=re.M)
    a, b = fonte.splitlines(), copia.splitlines()
    ops = [op for op in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op[0] != "equal"]
    mudadas = [(a[i1], b[j1]) for _, i1, i2, j1, j2 in ops if i2 - i1 == 1 and j2 - j1 == 1]
    verificar(n == 1 and len(ops) == 1 and len(mudadas) == 1 and mudadas[0][0].startswith("N_RUNS = 5"),
              f"a cópia do detect_vulnerabilities.py só difere numa linha: {mudadas}")
    destino = TRAB / "detect_vulnerabilities.py"
    destino.write_text(copia, encoding="utf-8")
    return destino


def treinar(runs=5):
    secao(f"3. detect_vulnerabilities.py {'sem alterações' if runs == 5 else f'com N_RUNS = {runs} (o resto sem alterações)'}"
          f" (modo corrigido, lido do interruptor)")
    script = script_de_treino(runs)
    # PYTHONPATH: a cópia (fora de classification/) importa load_datasets, load_graphcodebert_embeddings e
    # detect_vulnerabilities_vgg do repositório, como o original
    env = dict(os.environ, EMBEDDING_DIR=str(EMB), MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join([str(CLASSIFICACAO)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    env.pop("LEGACY_EDGE_ALIGNMENT", None)
    t = time.time()
    with open(TRAB / "treino.log", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, "-X", "utf8", str(script)],
                           cwd=CWD, env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"  terminou com código {r.returncode} em {(time.time() - t) / 60:.1f} min (log: {TRAB / 'treino.log'})")
    verificar(r.returncode == 0, "o treino corre até ao fim sem erros")


def verificar_resultados():
    secao("4. Verificações depois do treino")
    log = (TRAB / "treino.log").read_text(encoding="utf-8", errors="replace")
    verificar("Edge alignment: fixed" in log, "o treino correu no modo corrigido (linha 'Edge alignment: fixed' no log)")
    verificar("datasets_embeddings/pdg-graphcodebert-embeddings-aligned.pkl" in log
              and "Loading cached GraphCodeBERT embeddings from datasets_embeddings/pdg-graphcodebert-embeddings.pkl" not in log,
              "a cache antiga (desalinhada) não foi lida; foi criada a cache -aligned")
    antiga = CWD / "datasets_embeddings" / "pdg-graphcodebert-embeddings.pkl"
    verificar(sha(antiga) == (TRAB / "cache_antiga.sha").read_text(), "a cache antiga ficou intacta")

    with open(TRAB / "verdade.pkl", "rb") as f:
        verdade = pickle.load(f)
    df = pd.read_pickle(CWD / "datasets_embeddings" / "pdg-graphcodebert-embeddings-aligned.pkl")
    ficheiros = [f for f in os.listdir(EMB / "pdg") if f.endswith(".pt")]   # a mesma ordem que o loader usa
    esperados = [f for f in ficheiros if all(a in verdade[f]["textos"] and b in verdade[f]["textos"]
                                             for a, b in verdade[f]["arestas"])]
    verificar(len(df) == len(esperados), f"a cache -aligned tem os {len(esperados)} grafos esperados ({len(df)})")
    c = Counter()
    for (g, y), nome in zip(zip(df["graphs"], df["label"]), esperados):
        v = verdade[nome]
        dados = torch.load(EMB / "pdg" / nome)
        por_vetor = {}
        for nid, t in v["textos"].items():
            por_vetor.setdefault(dados["embeddings"][nid].numpy().tobytes(), set()).add(t)
        linha = [next(iter(s)) if len(s := por_vetor.get(x.tobytes(), set())) == 1 else None
                 for x in g.ndata["features"].numpy()]
        s, t = (x.tolist() for x in g.edges())
        reais = Counter((v["textos"][a], v["textos"][b]) for a, b in v["arestas"])
        vistas = Counter((linha[a], linha[b]) for a, b in zip(s, t))
        c["arestas"] += sum(reais.values())
        c["certas"] += sum((vistas & reais).values())
        c["grafos"] += 1
        c["grafos_certos"] += vistas == reais and None not in linha
        c["labels"] += int(y) == v["y"]
    verificar(c["certas"] == c["arestas"] and c["grafos_certos"] == c["grafos"],
              f"cache que o treino usou: {c['certas']}/{c['arestas']} arestas certas, {c['grafos_certos']}/{c['grafos']} grafos exatos")
    verificar(c["labels"] == c["grafos"], f"labels certas: {c['labels']}/{c['grafos']}")

    runs = CWD / "output_baseline" / "runs"
    pastas = sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []
    verificar(pastas and all("-aligned" in p.name for p in pastas),
              f"{len(pastas)} pastas de testes, todas com '-aligned' no nome")
    emb_files = list(runs.rglob("dgcnn_embeddings_test_after.pt")) if runs.exists() else []
    verificar(len(emb_files) >= 1, f"embeddings para o SVM/XGBoost gravadas em {len(emb_files)} runs")
    # as 4 configurações (T12, T10, T9, T11: VGG e Conv1D, com e sem autoencoder) chegaram ao fim
    resumos = list(runs.rglob("multi_run_summary.csv")) if runs.exists() else []
    configs = {cfg for cfg in ("T12", "T10", "T9", "T11") if f"RUN 1/" in log and f"({cfg})" in log}
    verificar(len(resumos) == 4 and configs == {"T12", "T10", "T9", "T11"},
              f"as 4 configurações correram até ao fim ({len(resumos)} multi_run_summary.csv; no log: {sorted(configs)})")
    for p in resumos:
        m = pd.read_csv(p)
        print(f"  {p.parent.name[:3]}  runs {len(m)}  precisão {m['precision'].mean():.3f}  recall {m['recall'].mean():.3f}"
              f"  F1 {m['f1_score'].mean():.3f}")
    for p in pastas:
        print(f"  {p.name}")
    for j in sorted(runs.rglob("*.json"))[:0]:
        print(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--so-verificar", action="store_true")
    ap.add_argument("--runs", type=int, default=5,
                    help="runs por configuração (5 = o script tal como está; menos = uma cópia que só muda N_RUNS)")
    a = ap.parse_args()
    cabecalho("01 - Ponta a ponta: detect_vulnerabilities.py sem alterações, com a correção")
    if not a.so_verificar:
        if TRAB.exists():
            import shutil
            shutil.rmtree(TRAB)
        preparar(a.n)
        TRAB.mkdir(parents=True, exist_ok=True)
        treinar(a.runs)
    verificar_resultados()
    print()
    if falhas:
        print(f"RESULTADO: {len(falhas)} VERIFICAÇÃO(ÕES) FALHARAM")
        for f in falhas:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULTADO: TODAS AS VERIFICAÇÕES PASSARAM")


if __name__ == "__main__":
    main()
