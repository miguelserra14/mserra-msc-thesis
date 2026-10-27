# 03 — Funções com o mesmo nome partilham o grafo e a etiqueta

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: BAIXA**

> Duas funções com o mesmo nome no mesmo ficheiro ficam ligadas ao mesmo `.dot`, e só a etiqueta da primeira sobrevive. O erro foi reproduzido, mas nos dados do Linux só afeta 61 funções e causa 1 conflito de etiqueta.

## Pergunta

Quando um ficheiro tem duas funções com o mesmo nome (por exemplo, nos dois ramos de um `#ifdef`), cada uma fica com o seu grafo e a sua etiqueta?

## Como o teste funciona

1. No ficheiro de teste há duas `helper` (linhas 24 e 26); no CSV de entrada a segunda está marcada vulnerável. O teste vê que `.dot` o mapeamento original atribuiu a cada uma e que etiqueta ficou no `.pt`.
2. Demonstra a correção, que mapeia por nome **e** linha de início.
3. Conta quantas funções do dataset real do Linux estão nesta situação.

## Resultado

No teste, com o código original:

```
.dot exportados com o nome 'helper': ['2-pdg.dot', '3-pdg.dot']  (linhas [24, 26])
helper  False  3-pdg.dot
helper  True   3-pdg.dot
.dot de 'helper' nunca usado: ['2-pdg.dot']
etiqueta guardada no .pt: {'3-pdg.pt': 0}   <- a 'helper' vulnerável perdeu a etiqueta 1
```

Com a correção (nome + linha), cada uma das 7 funções fica com o seu `.dot`, incluindo `helper` linhas 24 → `2-pdg.dot` e `helper` linhas 26–30 → `3-pdg.dot`.

Nos dados reais (`linux-functions.csv`): **61 de 110 853 funções (0,06%)** partilham commit, ficheiro e nome com outra; 3 delas são vulneráveis, e **1 grupo** tem etiquetas diferentes.

Uma observação lateral: o Joern exportou as duas `helper`, incluindo a do ramo `#ifdef CONFIG_FAST`, que não seria compilada. Ou seja, o Joern não avalia as condições do pré-processador.

## Porque acontece

[convert_dot_to_json.py:164-165](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/convert_dot_to_json.py#L164-L165) constrói um dicionário `nome → .dot`. Com nomes repetidos, a última entrada ganha (a ordem de `os.listdir` decide qual). As duas linhas do CSV recebem o mesmo caminho e, como o `create_embeddings.py` salta caminhos já processados, a etiqueta é a da primeira linha.

## Impacto nos resultados da tese

Pequeno: no máximo uma amostra com etiqueta errada e algumas dezenas de funções representadas pelo grafo da homónima. O valor do teste está em mostrar que o mapeamento só pelo nome não é fiável, o que pesa também no teste 05.

## Como resolver

Mapear pelo nome e pela linha de início. O dataset dá `Lines: [início, fim]` e a etiqueta do nó METHOD no `.dot` tem a linha (`METHOD, 26<BR/>helper`):

```python
import re

def linha_do_metodo(caminho_dot):
    with open(caminho_dot, encoding="utf-8", errors="ignore") as f:
        m = re.search(r'label = <METHOD, (\d+)<BR/>', f.read())
    return int(m.group(1)) if m else None

# em map_single_commit_to_graph, em vez de {nome: dot}:
indice = {}
for graph_file in os.listdir(graph_output_directory):
    if graph_file.endswith(".dot"):
        caminho = os.path.join(graph_output_directory, graph_file)
        indice.setdefault(get_graph_name(graph_output_directory, graph_file), []).append((linha_do_metodo(caminho), caminho))

for function in functions:
    inicio, fim = function["Lines"]
    candidatos = [c for linha, c in indice.get(function["Name"], []) if linha is not None and inicio <= linha <= fim]
    graph_filepath = candidatos[0] if len(candidatos) == 1 else ""   # 0 ou 2+ candidatos: não mapear
```

## Limitações desta verificação

A contagem nos dados reais usa o CSV de origem; não verifica quais destas 61 funções acabaram com grafo.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 03_funcoes_mesmo_nome\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
