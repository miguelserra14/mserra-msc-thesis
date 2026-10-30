"""Diagnóstico dos grafos sintéticos de teste_aprofundado.py: tamanhos, m vs etiqueta, pesos da sonda e colapso da DGCNN."""
import random, statistics as st, sys, time
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import teste_aprofundado as A
import numpy as np
import torch, torch.nn as nn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

dados = A.dados_sinteticos(150)
chave = lambda k: "->".join(k) if isinstance(k, tuple) else k

for (r, t), lista in dados.items():
    print(f"\n=== {r} {A.NOMES_TAREFA[t]}: {len(lista)} grafos; classe 1 = {sum(x[3] for x in lista)}")
    nos = [x[1].num_nodes() for x in lista]; ar = [x[1].num_edges() for x in lista]
    print(f"nós: mín {min(nos)} mediana {st.median(nos)} máx {max(nos)}; arestas: mediana {st.median(ar)}; >32 nós: {sum(n > 32 for n in nos)}")
    print(f"isolados no grafo corrigido: {sum(any(d == 0 for d in (x[1].in_degrees() + x[1].out_degrees()).tolist()) for x in lista)} grafos")
    m0 = [x[5] for x in lista if x[3] == 0]; m1 = [x[5] for x in lista if x[3] == 1]
    print(f"m médio por classe: {np.mean(m0):.2f} vs {np.mean(m1):.2f}")
    for nome, i in (("corrigido", 1), ("com erro", 0)):
        S = [{chave(k): v for k, v in x[4][i].items()} for x in lista]
        v = DictVectorizer(sparse=False); X = v.fit_transform(S); y = np.array([x[3] for x in lista])
        c = LogisticRegression(max_iter=5000).fit(X, y).coef_[0]
        ordem = np.argsort(-abs(c))[:5]
        print(f"  {nome}: {X.shape[1]} tipos de par; maiores pesos: " + ", ".join(f"{v.feature_names_[j]} {c[j]:+.2f}" for j in ordem))

# Par verificação/memcpy visto por aresta (só tarefa estrutura)
for r in ("cfg", "pdg"):
    print(f"\nPares check/memcpy por classe, {r} estrutura (nº de grafos com pelo menos 1 aresta do par)")
    for nome, i in (("corrigido", 1), ("com erro", 0)):
        for par in (("check", "memcpy"), ("memcpy", "check")):
            cont = [sum(1 for x in dados[(r, "e")] if x[3] == y and x[4][i].get(par, 0) > 0) for y in (0, 1)]
            print(f"  {nome:10} {par[0]}->{par[1]}: classe0 {cont[0]}/150, classe1 {cont[1]}/150")

# Diagnóstico da DGCNN: seed 0, exatidão no treino, classes previstas no teste
print("\n=== DGCNN, seed 0: treino / teste / previsões da classe 1 no teste", flush=True)
CFG = A.CFG_DV
def corre(tr, te, seed=0):
    torch.manual_seed(seed)
    hd, k = CFG["hidden_dimension"], CFG["k_sortpooling"]
    enc = A.MODELO["DGCNNEncoder"](768, hd, k_sortpooling=k, pooling_type="sort_pooling", use_pooling=True)
    clf = A.MODELO["DGCNNConv1DClassifier"](input_dim=k * hd[-1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=CFG["learning_rate"])
    perda = nn.CrossEntropyLoss(); rng = random.Random(seed); bs = CFG["batch_size"]
    perdas = []
    enc.train(); clf.train()
    for ep in range(CFG["num_epochs"]):
        idx = list(range(len(tr))); rng.shuffle(idx); tot = 0
        for i in range(0, len(idx), bs):
            lote = [tr[j] for j in idx[i:i + bs]]
            l = perda(clf(enc([g for g, _ in lote])), torch.tensor([y for _, y in lote]))
            opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
        perdas.append(tot / (len(idx) / bs))
    enc.eval(); clf.eval()
    with torch.no_grad():
        ptr = clf(enc([g for g, _ in tr])).argmax(1).tolist()
        pte = clf(enc([g for g, _ in te])).argmax(1).tolist()
        h = enc([g for g, _ in te])
    atr = sum(p == y for p, (_, y) in zip(ptr, tr)) / len(tr)
    ate = sum(p == y for p, (_, y) in zip(pte, te)) / len(te)
    return atr, ate, sum(pte), len(te), perdas[0], perdas[-1], (h == 0).float().mean().item()

for (r, t), lista in dados.items():
    corte = int(.7 * len(lista))
    print(f"{r} {A.NOMES_TAREFA[t]}: teste classe1={sum(x[3] for x in lista[corte:])}/{len(lista) - corte}", flush=True)
    for nome, col in (("corrigido", 1), ("com erro", 0), ("sem arestas", 2)):
        t0 = time.time()
        a = corre([(x[col], x[3]) for x in lista[:corte]], [(x[col], x[3]) for x in lista[corte:]])
        print(f"  {nome:12} treino {a[0] * 100:5.1f}%  teste {a[1] * 100:5.1f}%  prevê classe1 {a[2]}/{a[3]}  perda {a[4]:.3f}->{a[5]:.3f}  ativação nula {a[6] * 100:.0f}%  ({time.time() - t0:.0f}s)", flush=True)
