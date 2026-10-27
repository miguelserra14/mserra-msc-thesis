# 12 — O SVM aprende com embeddings que o encoder já memorizou

**Veredicto: PROBLEMA CONFIRMADO (metodológico; impacto nos números da tese não medido) · Gravidade: MÉDIA**

> O SVM é treinado com as embeddings dos grafos de treino, produzidas por um encoder que foi treinado nesses mesmos grafos, e é testado com embeddings de grafos que o encoder nunca viu. Com etiquetas aleatórias, as embeddings de treino permitem prever a etiqueta com 100% de accuracy, e as de teste 48%.

## Pergunta

As features que o SVM usa no treino e no teste vêm da mesma distribuição?

## Como o teste funciona

1. **Parte estática:** confirma de onde vêm `X_train` e `X_test` do SVM.
2. **Demonstração do mecanismo:** com as classes **originais** `DGCNNEncoder` e `DGCNNConv1DClassifier` e os hiperparâmetros do script, treina em 300 grafos sintéticos com **etiquetas aleatórias** (não há nada de real para aprender) durante 15 épocas. Depois mede, por validação cruzada 5-fold **dentro** de cada conjunto, se as embeddings permitem prever a etiqueta. Um encoder não treinado serve de controlo.

## Resultado

**Origem dos dados:**
- [detect_vulnerabilities.py:1038](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L1038): `save_embeddings(dataset=trainset, prefix='train_after')` com o encoder já treinado nesse `trainset`;
- [new_classifiers_svm_xgboost.py:54](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/new_classifiers_svm_xgboost.py#L54): `X_train = dgcnn_embeddings_train_after.pt`; linha 58: `X_test = dgcnn_embeddings_test_after.pt`.

**Demonstração** (accuracy por validação cruzada; o acaso é 0,50):

| Embeddings | Accuracy |
|---|---|
| treino, encoder treinado nelas | **1,00** |
| teste, mesmo encoder | 0,48 |
| treino, encoder não treinado (controlo) | 0,50 |

Um SVM treinado nas embeddings de treino e avaliado nas de teste acerta 0,49.

## Porque acontece

Durante o treino, o encoder ajusta-se às etiquetas dos grafos de treino, incluindo ao ruído. As embeddings desses grafos ficam separadas por classe de uma forma que **não existe** nos grafos novos. O SVM aprende uma fronteira nessa separação artificial e depois é aplicado a embeddings que não a têm.

Em `new_classifiers_svm_xgboost.py`, as projeções PCA "antes/depois do SMOTETomek" são feitas com as embeddings de treino, que são exatamente as que têm esta separação. O SMOTETomek também cria positivos sintéticos nessa zona.

## Impacto nos resultados da tese

Com dados reais, o encoder aprende alguma coisa genuína e alguma coisa memorizada. O SVM herda a parte memorizada como se fosse sinal, o que tende a piorar a generalização e a dar uma imagem enganadora da separação entre classes nos gráficos. **A magnitude não foi medida**: seriam precisas as embeddings reais de treino e teste.

## Como resolver

Treinar o SVM com embeddings **fora da amostra**:

1. **K-fold sobre o treino:** treinar o encoder em K−1 partes, gerar as embeddings da parte de fora e repetir para todas as partes. Assim obtém-se uma embedding fora da amostra para cada grafo de treino; o SVM treina com essas. Para o teste, usar um encoder treinado em todo o treino.
2. **Mais simples:** dividir o treino em A e B. O encoder treina em A e o SVM treina com as embeddings de B.
3. Ou abdicar do SVM e reportar diretamente o classificador da DGCNN, que já é treinado de ponta a ponta.

Os gráficos PCA devem mostrar as embeddings de teste (ou de validação), não as de treino.

## Limitações desta verificação

A demonstração usa grafos sintéticos com etiquetas aleatórias para isolar o mecanismo. Prova que o efeito existe com este código, não quanto pesa nos resultados da tese.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 12_svm_sobre_embeddings_de_treino\teste.py
```

Não precisa da preparação. Demora cerca de 1,5 min (treina a DGCNN em CPU). A saída completa está em [resultado.txt](resultado.txt).
