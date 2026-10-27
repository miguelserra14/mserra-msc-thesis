# 14 — Arquitetura DGCNN: só usa a última camada e duplica colunas

**Veredicto: PROBLEMA CONFIRMADO — desvio de desenho, não erro · Gravidade: BAIXA**

> As dimensões batem certo e os dois caminhos de classificação (Conv1D e VGG) correm. Mas o encoder só usa a saída da última camada GraphConv, ao contrário do DGCNN original. O adaptador da VGG produz 128 colunas, das quais só 32 são distintas.

## Pergunta

A arquitetura faz o que o relatório descreve, e os tensores têm as dimensões esperadas?

## Como o teste funciona

Corre as classes **originais** de [detect_vulnerabilities.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py) (`DGCNNEncoder`, `DGCNNConv1DClassifier`, `DGCNNVGGAdapter`, `adjust_to_vgg`, `VGGnet`), carregadas sem executar o resto do script e com os hiperparâmetros do script. Usa como entrada os 6 grafos PDG da preparação, lidos com o loader original. Mede as formas dos tensores, conta as colunas distintas à saída do adaptador e verifica que camadas entram na representação.

## Resultado

| Caminho | Etapa | Forma |
|---|---|---|
| Conv1D (T9/T11) | encoder com SortPooling (k=32) | (6, 1024) ✔ |
| | classificador | (6, 2) ✔ |
| VGG (T10/T12) | features por nó à saída do encoder | 32 |
| | `DGCNNVGGAdapter` | (6, 32, 32, **128**), com **32 colunas distintas em 128** |
| | `adjust_to_vgg` | (6, 32, 224, 224) ✔ |
| | `VGGnet` | (6, 2) ✔ |

Em `DGCNNEncoder.forward`, a única chamada `batch_node_features.append(h)` ([linha 182](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L182)) acontece depois de `conv4`.

## Porque acontece

- O DGCNN original (Zhang et al., 2018) concatena as saídas de todas as camadas de convolução (aqui seriam 4 × 32 = 128 por nó) e usa `tanh`. O encoder deste código guarda só a última camada (32), com ReLU ([linhas 175-182](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L175-L182)).
- O adaptador VGG é criado com `embedding_dim=sum(hidden_dimension)` = 128 ([linha 796](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L796)), o que indica que a concatenação estava prevista. Como chegam só 32 colunas, o `AdaptiveMaxPool2d((32, 128))` ([linha 224](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L224)) repete cada coluna 4 vezes.

## Impacto nos resultados da tese

- Perde-se a informação das camadas intermédias (vizinhança a 1, 2 e 3 saltos), que o DGCNN usa para descrever estrutura a várias escalas. A DGCNN descrita na tese não é exatamente a implementada.
- No caminho VGG, 3/4 da "imagem" são cópias, o que só gasta memória e tempo.
- Não invalida resultados, mas deve ser corrigido ou descrito.

## Como resolver

Concatenar as camadas, como no DGCNN original:

```python
h1 = torch.tanh(self.conv1(g, h))
h2 = torch.tanh(self.conv2(g, h1))
h3 = torch.tanh(self.conv3(g, h2))
h4 = torch.tanh(self.conv4(g, h3))
batch_node_features.append(torch.cat([h1, h2, h3, h4], dim=1))   # 128 por nó
```

E ajustar as dimensões que dependem disso: `DGCNNConv1DClassifier(input_dim=k_sortpooling * sum(hidden_dimension))`, e o `GraphDecoder` do autoencoder, se continuar a ser usado. O adaptador VGG já espera 128.

Nota: o SortPooling do DGL ordena os nós pelo **último** canal. Com a concatenação, esse canal passa a ser o último da `conv4`, que é o mesmo comportamento do DGCNN original.

## Limitações desta verificação

Feita com 6 grafos pequenos. Só verifica formas e fluxo de dados, não a qualidade do treino.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 14_arquitetura_dgcnn\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
