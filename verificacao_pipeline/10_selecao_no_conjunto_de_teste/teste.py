"""10 - Alguma decisão (hiperparâmetros, escolha de run) é tomada olhando para o conjunto de teste?

Verificação estática (árvore sintática do Python, sem executar os scripts) de
new_classifiers_svm_xgboost.py e detect_vulnerabilities.py, mais o texto do relatório final
(dados/relatorio_constanca.txt, extraído do PDF com pdftotext).
"""
import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import CLASSIFICACAO, DADOS, atribuicoes_de_topo, cabecalho, ler_ast, secao, veredicto


def nomes_usados(no):
    return {n.id for n in ast.walk(no) if isinstance(n, ast.Name)}


def main():
    cabecalho("10 - Decisões tomadas com o conjunto de teste")
    svm = CLASSIFICACAO / "new_classifiers_svm_xgboost.py"
    dv = CLASSIFICACAO / "detect_vulnerabilities.py"

    secao("Optuna (new_classifiers_svm_xgboost.py)")
    arvore = ler_ast(svm)
    objetivo = next(n for n in ast.walk(arvore) if isinstance(n, ast.FunctionDef) and n.name == "objective")
    usados = nomes_usados(objetivo)
    print(f"função objective (linha {objetivo.lineno}) usa: {sorted(usados & {'X_test', 'y_test', 'X_train_resampled', 'y_train_resampled'})}")
    retorno = next(n for n in ast.walk(objetivo) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "score" for t in n.targets))
    print(f"   score = {ast.unparse(retorno.value)}   (linha {retorno.lineno})")
    otimiza_no_teste = {"X_test", "y_test"} <= usados
    todos_nomes = nomes_usados(arvore)
    validacao = sorted(n for n in todos_nomes if re.search(r"val", n, re.I) and n not in {"evaluate_model"})
    print(f"variáveis de validação no script: {validacao or 'nenhuma'}")
    data_dir = atribuicoes_de_topo(svm).get("DATA_DIR", "")
    print(f"DATA_DIR (embeddings usadas pelo SVM): ...{data_dir[-60:]}")

    secao("Treino DGCNN (detect_vulnerabilities.py)")
    arvore_dv = ler_ast(dv)
    splits = [n for n in ast.walk(arvore_dv) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "train_test_split"]
    print(f"chamadas a train_test_split: {len(splits)} (linha {splits[0].lineno}) -> só treino/teste, sem validação")
    print("as métricas por run são calculadas no testset e guardadas em multi_run_summary.csv")

    secao("Relatório final (como os números foram escolhidos)")
    texto = re.sub(r"\s+", " ", (DADOS / "relatorio_constanca.txt").read_text(encoding="utf-8"))
    frases = {
        "Tabela 5.3 usa a melhor run em F1": "contain the best run results in terms of F1-Score",
        "SVM com Optuna, T1": "AST with SortPooling (T1): C ≈ 0.00315",
        "SVM com Optuna, T7": "AST with AMP (T7): C ≈ 0.00582",
        "SVM com Optuna, T11": "PDG with AMP (T11): C ≈ 0.00525",
    }
    encontradas = {}
    for descricao, frase in frases.items():
        encontradas[descricao] = frase in texto
        print(f"{descricao:36} '{frase}': {'encontrado' if encontradas[descricao] else 'NÃO encontrado'}")

    problema = otimiza_no_teste and encontradas["Tabela 5.3 usa a melhor run em F1"]
    veredicto(problema, "ALTA",
              "o Optuna escolhe C, gamma e kernel pelo F1 no conjunto de teste, a Tabela 5.3 apresenta a melhor de 5 "
              "runs escolhida no teste, e não existe conjunto de validação. As métricas finais ficam enviesadas para "
              "cima e não são estimativas independentes.")


if __name__ == "__main__":
    main()
