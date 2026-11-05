"""Treina o caminho SortPooling da tese (T9 -> SMOTETomek -> SVM/XGBoost, Tabela 5.7) numa só condição.

Condições, que só diferem nas arestas (mesmos grafos, features, divisão treino/teste e inicialização):
  erro         edge_index de parse_dot_edges + loader, ambos originais
  corrigido    uma só ordem de nós para features e arestas (a correção proposta)
  sem_arestas  sem arestas: mede quanto a estrutura contribui

Classes originais e hiperparâmetros lidos de detect_vulnerabilities.py. Só entram os grafos que o loader original
consegue carregar, como na tese. Grava um .json em _gerado/impacto_tese/resultados/ (retomável).

Uso: python -X utf8 treinar.py --tipo pdg --condicao erro --seed 42
"""
import argparse
import ctypes
import json
import pickle
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from comum import CLASSIFICACAO, GERADO, atribuicoes_de_topo, carregar_definicoes

import dgl
import numpy as np
import torch
import torch.nn as nn
from imblearn.combine import SMOTETomek
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.insert(0, str(CLASSIFICACAO))
from load_graphcodebert_embeddings import create_dgl_graph_from_embedding  # noqa: E402  (código original)

SAIDA = GERADO / "impacto_tese"
DV = CLASSIFICACAO / "detect_vulnerabilities.py"
PREVALENCIA_TESE = 0.026


def carregar_grafos(tipo, condicao):
    """Só guarda em cada grafo os índices dos vetores; as features são colocadas lote a lote, para poupar RAM."""
    with open(SAIDA / "textos.pkl", "rb") as f:
        indice = {t: i for i, t in enumerate(pickle.load(f))}
    grafos, ys, perdidos = [], [], 0
    for p in sorted((SAIDA / "grafos").glob("*.pkl")):
        with open(p, "rb") as f:
            registo = pickle.load(f)
        for r in registo["grafos"]:
            if r["tipo"] != tipo:
                continue
            emb = {nid: torch.tensor([float(indice[t])], dtype=torch.float64) for nid, t in r["nos"]}
            if not emb:
                perdidos += 1
                continue
            d = {"embeddings": emb, "edge_index": torch.tensor(r["edge_index"], dtype=torch.long).reshape(2, -1),
                 "node_to_idx": r["node_to_idx"]}
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                g = create_dgl_graph_from_embedding(d)       # loader original (com o erro)
            if g is None:
                perdidos += 1                                 # também se perde na tese
                continue
            ordem = sorted(emb, key=int)
            ids = torch.tensor([indice[t] for t in (dict(r["nos"])[n] for n in ordem)], dtype=torch.long)
            if condicao == "corrigido":
                idx = {n: i for i, n in enumerate(ordem)}
                pares = [(idx[a], idx[b]) for a, b in r["arestas"] if a in idx and b in idx]
                g = dgl.graph(([a for a, _ in pares], [b for _, b in pares]), num_nodes=len(ordem))
            elif condicao == "sem_arestas":
                g = dgl.graph(([], []), num_nodes=len(ordem))
            elif condicao != "erro":
                raise ValueError(condicao)
            g.ndata.pop("features", None)
            g.ndata["ids"] = ids
            grafos.append(g)
            ys.append(r["y"])
    return grafos, np.array(ys), perdidos


def com_features(grafos, vetores):
    for g in grafos:
        g.ndata["features"] = torch.from_numpy(np.array(vetores[g.ndata["ids"].numpy()]))
    return grafos


def sem_features(grafos):
    for g in grafos:
        g.ndata.pop("features", None)


def metricas(y, pred, score):
    rec = recall_score(y, pred, zero_division=0)
    fpr = float(((pred == 1) & (y == 0)).sum() / max(1, (y == 0).sum()))
    pi = PREVALENCIA_TESE
    denom = rec * pi + fpr * (1 - pi)
    prec_aj = rec * pi / denom if denom else 0.0
    return {"precisao": precision_score(y, pred, zero_division=0), "recall": rec, "f1": f1_score(y, pred, zero_division=0),
            "fpr": fpr, "roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score),
            "precisao_aj_2.6": prec_aj, "f1_aj_2.6": 2 * prec_aj * rec / (prec_aj + rec) if prec_aj + rec else 0.0,
            "fracao_positivos_previstos": float((pred == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", required=True)
    ap.add_argument("--condicao", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--epocas", type=int)
    ap.add_argument("--max-grafos", type=int)
    a = ap.parse_args()
    destino = SAIDA / "resultados" / f"{a.tipo}_{a.condicao}_{a.seed}.json"
    teste_fumo = a.max_grafos is not None or a.epocas is not None
    if destino.exists() and not teste_fumo:
        print("já existe", destino.name)
        return
    destino.parent.mkdir(exist_ok=True)
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    torch.set_num_threads(a.threads)

    cfg = atribuicoes_de_topo(DV)
    mod = carregar_definicoes(DV, {"DGCNNEncoder", "DGCNNConv1DClassifier"}, {"dropout_rate": cfg["dropout_rate"]})
    vetores = np.load(SAIDA / "vetores.npy", mmap_mode="r")
    grafos, y, perdidos = carregar_grafos(a.tipo, a.condicao)
    if a.max_grafos:
        grafos, y = grafos[:a.max_grafos], y[:a.max_grafos]
    print(f"{a.tipo}/{a.condicao}/seed {a.seed}: {len(grafos)} grafos, {y.sum()} vulneráveis, {perdidos} perdidos no loader", flush=True)

    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=a.seed)
    tr = [grafos[i] for i in idx_tr]
    te = [grafos[i] for i in idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    hd, k, bs = cfg["hidden_dimension"], cfg["k_sortpooling"], cfg["batch_size"]
    enc = mod["DGCNNEncoder"](768, hd, k_sortpooling=k, pooling_type="sort_pooling", use_pooling=True)
    clf = mod["DGCNNConv1DClassifier"](input_dim=k * hd[-1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=cfg["learning_rate"])
    contagens = np.bincount(y_tr)
    pesos = torch.tensor([len(y_tr) / (2 * c) for c in contagens], dtype=torch.float)   # AUTO_WEIGHTING
    perda = nn.CrossEntropyLoss(weight=pesos)
    epocas = a.epocas or cfg["num_epochs"]

    t0 = time.time()
    historico = []
    enc.train()
    clf.train()
    for ep in range(epocas):
        total = n = 0
        for i in range(0, len(tr), bs):                      # sem baralhar, como o DataLoader da tese
            lote = com_features(tr[i:i + bs], vetores)
            saida = clf(enc(lote))
            l = perda(saida, torch.tensor(y_tr[i:i + bs], dtype=torch.long))
            opt.zero_grad()
            l.backward()
            opt.step()
            total += l.item()
            n += 1
            sem_features(lote)
        historico.append(total / n)
        print(f"época {ep + 1}/{epocas} perda {historico[-1]:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)

    enc.eval()
    clf.eval()

    def extrair(conjunto):
        zs, ps = [], []
        with torch.no_grad():
            for i in range(0, len(conjunto), bs):
                lote = com_features(conjunto[i:i + bs], vetores)
                z = enc(lote)
                zs.append(z)
                ps.append(torch.softmax(clf(z), dim=1)[:, 1])
                sem_features(lote)
        return torch.cat(zs).numpy(), torch.cat(ps).numpy()

    z_tr, p_tr = extrair(tr)
    z_te, p_te = extrair(te)
    res = {"tipo": a.tipo, "condicao": a.condicao, "seed": a.seed, "n": len(y), "n_vuln": int(y.sum()), "perdidos_loader": perdidos,
           "n_teste": len(y_te), "n_vuln_teste": int(y_te.sum()), "min_treino": (time.time() - t0) / 60, "perda": historico,
           "exatidao_treino": float(((p_tr > .5).astype(int) == y_tr).mean()),
           "conv1d": metricas(y_te, (p_te > .5).astype(int), p_te)}

    z_bal, y_bal = SMOTETomek(random_state=42).fit_resample(z_tr, y_tr)              # como new_classifiers_svm_xgboost.py
    svm = SVC(kernel="rbf", random_state=42).fit(z_bal, y_bal)
    res["svm"] = metricas(y_te, svm.predict(z_te), svm.decision_function(z_te))
    xgb = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=a.threads).fit(z_bal, y_bal)
    res["xgb"] = metricas(y_te, xgb.predict(z_te), xgb.predict_proba(z_te)[:, 1])
    if teste_fumo:
        print(json.dumps({k: res[k] for k in ("n", "n_vuln", "perdidos_loader", "min_treino", "exatidao_treino", "conv1d", "svm", "xgb")}, indent=1))
        return
    destino.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("FIM", destino.name, flush=True)


if __name__ == "__main__":
    main()
