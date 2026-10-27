"""12 - O SVM é treinado e testado com embeddings comparáveis?

Parte 1 (estática): confirma que o SVM treina com as embeddings do conjunto de TREINO produzidas pelo
encoder que foi treinado nesse mesmo conjunto, e testa com embeddings de amostras que o encoder nunca viu.
Parte 2 (demonstração do mecanismo, dados sintéticos): usa as classes ORIGINAIS DGCNNEncoder e
DGCNNConv1DClassifier com etiquetas ALEATÓRIAS (não há nada para aprender). Se o SVM separar bem as
embeddings de treino, essa separação é memorização do encoder e não generaliza.
"""
import ast
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import CLASSIFICACAO, atribuicoes_de_topo, cabecalho, carregar_definicoes, ler_ast, secao, veredicto

import dgl
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

DV = CLASSIFICACAO / "detect_vulnerabilities.py"
SVM = CLASSIFICACAO / "new_classifiers_svm_xgboost.py"


def parte_estatica():
    secao("Parte 1 - de onde vêm as embeddings do SVM")
    for n in ast.walk(ler_ast(DV)):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "save_embeddings":
            kws = {k.arg: ast.unparse(k.value) for k in n.keywords}
            print(f"detect_vulnerabilities.py:{n.lineno} save_embeddings(dataset={kws['dataset']}, prefix={kws['prefix']}) "
                  "com o encoder já treinado nesse trainset")
    for n in ast.walk(ler_ast(SVM)):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"X_train", "X_test"}:
            print(f"new_classifiers_svm_xgboost.py:{n.lineno} {n.targets[0].id} = ...{ast.unparse(n.value)[-60:]}")


def grafos_sinteticos(n, gerador):
    grafos = []
    for _ in range(n):
        nos = gerador.randint(6, 14)
        src = [gerador.randrange(nos) for _ in range(nos * 2)]
        dst = [gerador.randrange(nos) for _ in range(nos * 2)]
        g = dgl.graph((src, dst), num_nodes=nos)
        g.ndata["features"] = torch.randn(nos, 768)
        grafos.append(g)
    return grafos


def parte_demonstracao():
    secao("Parte 2 - demonstração com etiquetas aleatórias (classes originais)")
    cfg = atribuicoes_de_topo(DV)
    defs = carregar_definicoes(DV, {"DGCNNEncoder", "DGCNNConv1DClassifier"}, {"dropout_rate": cfg["dropout_rate"]})
    torch.manual_seed(0)
    gerador = random.Random(0)
    treino, teste = grafos_sinteticos(300, gerador), grafos_sinteticos(300, gerador)
    y_tr = torch.tensor([gerador.randint(0, 1) for _ in treino])
    y_te = np.array([gerador.randint(0, 1) for _ in teste])

    enc = defs["DGCNNEncoder"](768, cfg["hidden_dimension"], k_sortpooling=cfg["k_sortpooling"])
    clf = defs["DGCNNConv1DClassifier"](input_dim=cfg["k_sortpooling"] * cfg["hidden_dimension"][-1])
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()), lr=cfg["learning_rate"])
    perda = nn.CrossEntropyLoss()
    for epoca in range(15):
        for i in range(0, len(treino), cfg["batch_size"]):
            loss = perda(clf(enc(treino[i:i + cfg["batch_size"]])), y_tr[i:i + cfg["batch_size"]])
            opt.zero_grad(); loss.backward(); opt.step()

    def embeddings(encoder, grafos):
        encoder.eval()
        with torch.no_grad():
            return torch.cat([encoder(grafos[i:i + 50]) for i in range(0, len(grafos), 50)]).numpy()

    controlo = defs["DGCNNEncoder"](768, cfg["hidden_dimension"], k_sortpooling=cfg["k_sortpooling"])  # não treinado
    e_tr, e_te = embeddings(enc, treino), embeddings(enc, teste)
    # Validação cruzada DENTRO de cada conjunto: mede se as embeddings "contêm" a etiqueta aleatória
    cv_tr = cross_val_score(SVC(kernel="rbf"), e_tr, y_tr.numpy(), cv=5).mean()
    cv_te = cross_val_score(SVC(kernel="rbf"), e_te, y_te, cv=5).mean()
    cv_ctrl = cross_val_score(SVC(kernel="rbf"), embeddings(controlo, treino), y_tr.numpy(), cv=5).mean()
    svm = SVC(kernel="rbf").fit(e_tr, y_tr.numpy())
    acc_te = (svm.predict(e_te) == y_te).mean()
    print("accuracy (validação cruzada 5-fold dentro de cada conjunto; acaso = 0.50):")
    print(f"   embeddings de treino, encoder treinado nelas:   {cv_tr:.2f}")
    print(f"   embeddings de teste, mesmo encoder:             {cv_te:.2f}")
    print(f"   embeddings de treino, encoder NÃO treinado:     {cv_ctrl:.2f}  (controlo)")
    print(f"SVM treinado nas embeddings de treino e avaliado nas de teste: {acc_te:.2f}")
    return cv_tr, cv_te, cv_ctrl


def main():
    cabecalho("12 - SVM treinado com embeddings de treino (in-sample) e testado com embeddings out-of-sample")
    parte_estatica()
    cv_tr, cv_te, cv_ctrl = parte_demonstracao()
    print("\nO SVM aprende uma fronteira numa distribuição (treino memorizado pelo encoder) diferente daquela em")
    print("que é avaliado (teste). Em new_classifiers_svm_xgboost.py, as projeções PCA 'antes/depois do SMOTETomek'")
    print("são feitas com as embeddings de treino, onde esta separação artificial aparece.")
    veredicto(cv_tr > max(cv_te, cv_ctrl) + 0.1, "MÉDIA",
              f"confirmado no código. Com etiquetas aleatórias, as embeddings de treino deixam prever a etiqueta com "
              f"{cv_tr:.0%} de accuracy e as de teste com {cv_te:.0%}: o SVM aprende sobre memorização do encoder. "
              "O impacto nos números da tese não está medido (faltam as embeddings reais).")


if __name__ == "__main__":
    main()
