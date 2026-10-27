# 15 — Pontos verificados que estão corretos

**Veredicto: SEM PROBLEMA · Gravidade: NENHUMA**

> 15 de 15 verificações OK. Estes passos da pipeline foram auditados e não precisam de correção.

## Para que serve

Delimitar o que foi auditado. Os problemas estão nos testes 01–14; os pontos abaixo foram verificados e estão bem. Se a pipeline for alterada, este teste confirma que nada aqui deixou de funcionar.

## Verificações

### Extração e embeddings (executado sobre a preparação, para AST, CFG e PDG)

| Verificação | Resultado |
|---|---|
| A etiqueta `True`/`False` do CSV chega ao `.pt` como 1/0 | OK ×3 |
| Linhas repetidas no CSV não geram `.pt` repetidos (14 linhas → 6 caminhos → 6 `.pt`) | OK ×3 |
| Todos os nós têm um vetor de 768 dimensões | OK ×3 |

### Treino da DGCNN (análise estática de `detect_vulnerabilities.py`)

| Verificação | Resultado |
|---|---|
| `save_embeddings` percorre os dados sem baralhar, por isso as embeddings ficam alinhadas com as etiquetas (linha 396) | OK |
| Os pesos de classe são calculados só com as etiquetas de treino (`np.bincount(train_labels)`, linha 654) | OK |
| A divisão é estratificada pela etiqueta (linha 640) | OK |

### SVM (análise estática de `new_classifiers_svm_xgboost.py`)

| Verificação | Resultado |
|---|---|
| O SMOTETomek é aplicado só ao treino (`smt.fit_resample(X_train, y_train)`, linha 68) | OK |
| O PCA dos gráficos é ajustado só com o treino (linha 76) | OK |
| O SVM final é treinado com o treino reamostrado, não com o teste (linha 261) | OK |

### Verificados noutros testes

| Ponto | Onde |
|---|---|
| O código multilinha não parte os `.txt`; `<BR/>` dentro de strings não estraga a limpeza das etiquetas | 02 |
| A numeração dos `.dot` do Joern é determinística | 05 |
| As dimensões SortPooling → Conv1D e AdaptiveMaxPool → VGG batem certo | 14 |

## O que este teste não cobre

- O código legado das features manuais do Luís (`graph_feature_extraction.py`, `reduce_graph*.py`, `load_datasets.py`, …). Não entra nos resultados com GraphCodeBERT: a secção 5.2 do relatório confirma que as tabelas usam "the new hybrid architecture, using GraphCodeBert embeddings".
- Os scripts de análise posterior (`analyze_embeddings.py`, `feature_correlation.py`, `create_heatmap_features.py`) e o exemplo `llm_code_classifier.py`.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 15_verificacoes_sem_problema\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
