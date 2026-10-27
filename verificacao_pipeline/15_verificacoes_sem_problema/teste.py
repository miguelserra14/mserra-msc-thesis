"""15 - Pontos da pipeline que foram verificados e estão corretos.

Cada verificação imprime OK ou FALHA. Serve para delimitar o que foi auditado: estes passos não
precisam de ser corrigidos.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import CLASSIFICACAO, PIPELINE, TIPOS, cabecalho, exigir_preparacao, ler_ast, secao, veredicto

import pandas as pd
import torch

DV = CLASSIFICACAO / "detect_vulnerabilities.py"
SVM = CLASSIFICACAO / "new_classifiers_svm_xgboost.py"
resultados = []


def check(descricao, ok, detalhe=""):
    resultados.append(ok)
    print(f"[{'OK   ' if ok else 'FALHA'}] {descricao}" + (f" - {detalhe}" if detalhe else ""))


def chamadas(arvore):
    return sorted((n for n in ast.walk(arvore) if isinstance(n, ast.Call)), key=lambda n: n.lineno)


def main():
    cabecalho("15 - Verificações em que não foi encontrado problema")
    exigir_preparacao()

    secao("Extração e embeddings (executado sobre a preparação)")
    for tipo in TIPOS:
        csv = pd.read_csv(PIPELINE / "output-data" / f"functions-{tipo}-linux-filtered.csv", sep=";")
        primeira = csv.drop_duplicates("CFG_filepath", keep="first").set_index("CFG_filepath")
        pts = list((PIPELINE / "dataset_final_embeddings" / tipo).glob("*.pt"))
        etiquetas_ok = all(
            int(torch.load(p, weights_only=False)["label"]) ==
            int(str(primeira.loc[torch.load(p, weights_only=False)["dot_path"], "vulnerable_label"]).lower() == "true")
            for p in pts)
        check(f"{tipo}: a etiqueta True/False do CSV chega ao .pt como 1/0", etiquetas_ok)
        check(f"{tipo}: linhas repetidas no CSV não geram .pt repetidos", len(pts) == csv.CFG_filepath.nunique(),
              f"{len(csv)} linhas, {csv.CFG_filepath.nunique()} caminhos, {len(pts)} .pt")
        dims = {tuple(v.shape) for p in pts for v in torch.load(p, weights_only=False)["embeddings"].values()}
        check(f"{tipo}: todos os nós têm um vetor de 768 dimensões", dims == {(768,)})
    print("(o código multilinha e strings com '<BR/>' também foram verificados no teste 02;")
    print(" a numeração determinística dos .dot no teste 05; as dimensões da DGCNN no teste 14)")

    secao("Treino da DGCNN (análise estática de detect_vulnerabilities.py)")
    arv = ler_ast(DV)
    se = next(n for n in ast.walk(arv) if isinstance(n, ast.FunctionDef) and n.name == "save_embeddings")
    dl = next(n for n in chamadas(se) if getattr(n.func, "id", "") == "DataLoader")
    kws = {k.arg for k in dl.keywords}
    check("save_embeddings percorre os dados sem baralhar (embeddings alinhadas com as etiquetas)",
          "shuffle" not in kws and "sampler" not in kws, f"linha {dl.lineno}")
    bincount = next(n for n in chamadas(arv) if ast.unparse(n.func) == "np.bincount")
    check("os pesos de classe são calculados só com as etiquetas de treino", ast.unparse(bincount.args[0]) == "train_labels",
          f"linha {bincount.lineno}: {ast.unparse(bincount)}")
    tts = next(n for n in chamadas(arv) if getattr(n.func, "id", "") == "train_test_split")
    kw = {k.arg: ast.unparse(k.value) for k in tts.keywords}
    check("a divisão é estratificada pela etiqueta", kw.get("stratify") == "run_df['label']", f"linha {tts.lineno}")

    secao("SVM (análise estática de new_classifiers_svm_xgboost.py)")
    arv = ler_ast(SVM)
    fr = next(n for n in chamadas(arv) if ast.unparse(n.func) == "smt.fit_resample")
    check("SMOTETomek é aplicado só ao treino", [ast.unparse(a) for a in fr.args] == ["X_train", "y_train"],
          f"linha {fr.lineno}: {ast.unparse(fr)}")
    pca = next(n for n in chamadas(arv) if ast.unparse(n.func) == "pca.fit_transform")
    check("o PCA dos gráficos é ajustado só com o treino", [ast.unparse(a) for a in pca.args] == ["X_train"],
          f"linha {pca.lineno}")
    fit_final = [n for n in chamadas(arv) if ast.unparse(n.func) == "best_svm_clf.fit"]
    check("o SVM final é treinado com o treino reamostrado (não com o teste)",
          bool(fit_final) and [ast.unparse(a) for a in fit_final[0].args] == ["X_train_resampled", "y_train_resampled"],
          f"linha {fit_final[0].lineno}" if fit_final else "")

    veredicto(False if all(resultados) else True, "NENHUMA",
              f"{sum(resultados)}/{len(resultados)} verificações OK. Estes passos estão corretos; os problemas da "
              "pipeline estão nos testes 01-14.")


if __name__ == "__main__":
    main()
