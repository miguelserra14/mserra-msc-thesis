# 10 — Decisões tomadas olhando para o conjunto de teste

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: ALTA**

> O Optuna escolhe os parâmetros do SVM pelo F1 no conjunto de teste. A Tabela 5.3 apresenta a melhor de 5 runs, escolhida no teste. Não existe conjunto de validação em nenhum ponto. As métricas finais ficam enviesadas para cima e não são estimativas independentes.

## Pergunta

Alguma escolha (hiperparâmetros, run a reportar, embeddings a usar) depende do desempenho no conjunto de teste?

## Como o teste funciona

Análise estática da árvore sintática do Python, sem executar os scripts:

- [new_classifiers_svm_xgboost.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/new_classifiers_svm_xgboost.py): que variáveis usa a função `objective` do Optuna, e se existe alguma variável de validação;
- [detect_vulnerabilities.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py): quantas divisões de dados existem.

Procura ainda, no texto do relatório final ([dados/relatorio_constanca.txt](../dados/relatorio_constanca.txt)), como foram escolhidos os números apresentados.

## Resultado

- A função `objective` ([linha 228](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/new_classifiers_svm_xgboost.py#L228)) usa `X_test` e `y_test`, e devolve `score = f1_score(y_test, y_pred)` ([linha 249](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/new_classifiers_svm_xgboost.py#L249)).
- Não há nenhuma variável de validação no script.
- O SVM usa as embeddings de uma run específica (`DATA_DIR = …/run_5/embeddings`).
- `detect_vulnerabilities.py` tem uma única divisão (treino/teste, [linha 640](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L640)).
- O relatório confirma o uso destes resultados:
  - "contain the best run results in terms of F1-Score" (Tabela 5.3);
  - configurações do Optuna para **T1** (C ≈ 0.00315), **T7** (C ≈ 0.00582) e **T11** (C ≈ 0.00525), apresentadas na Tabela 5.8 como as melhores.

## Porque acontece

Há três decisões tomadas com o teste:

1. **Hiperparâmetros do SVM:** o Optuna testa 10 combinações e fica com a de melhor F1 no teste. Depois o mesmo teste é usado para reportar o resultado.
2. **Escolha da run:** das 5 seeds de cada configuração, a Tabela 5.3 mostra a de melhor F1, em vez da média.
3. **Embeddings do SVM:** a run cujas embeddings alimentam o SVM foi escolhida entre as 5, ao que tudo indica pelo mesmo critério.

## Impacto nos resultados da tese

O efeito é sempre otimista: escolher o máximo de várias tentativas medidas no teste dá um número acima do desempenho esperado em dados novos. Com F1 abaixo de 0,15 e 2,6% de positivos, a diferença entre runs é sobretudo ruído, e a "melhor run" é em grande parte sorte. O mesmo vale para o ROC-AUC de 0,7468 apresentado para T1.

A magnitude do enviesamento não foi medida: faltam as embeddings e as predições reais. O `multi_run_summary.csv` de cada configuração já guarda as 5 runs e permite calcular a média.

## Como resolver

1. **Criar um conjunto de validação** a partir do treino (agrupado por função, ver 08), e usar só esse no Optuna:

```python
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
i_fit, i_val = next(gss.split(X_train, y_train, groups=grupos_train))

def objective(trial):
    clf = SVC(C=..., gamma=..., kernel=...)
    X_fit, y_fit = SMOTETomek(random_state=42).fit_resample(X_train[i_fit], y_train[i_fit])
    clf.fit(X_fit, y_fit)
    return f1_score(y_train[i_val], clf.predict(X_train[i_val]))   # nunca X_test
```

2. **Reportar média ± desvio padrão** das 5 runs (já calculados em `summary_stats.csv`), não a melhor.
3. Escolher a run e o número de épocas da DGCNN pela validação.
4. Usar o teste **uma só vez**, no fim, para os números finais da tese.

## Limitações desta verificação

A verificação é estática, mais o texto do relatório. Não mede quanto as métricas mudariam com uma validação correta.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 10_selecao_no_conjunto_de_teste\teste.py
```

Não precisa da preparação. A saída completa está em [resultado.txt](resultado.txt).
