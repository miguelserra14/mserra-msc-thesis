# 01 — As arestas ligam os nós errados

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: CRÍTICA**

> Só 15% das arestas que a DGCNN recebe ligam os nós certos. A forma do grafo mantém-se, mas o conteúdo dos nós fica trocado entre posições. A correção recupera 100%.

## Pergunta

O grafo DGL que entra na DGCNN liga os mesmos fragmentos de código que o `.dot` exportado pelo Joern?

## Como o teste funciona

1. Carrega cada `.pt` gerado pelo `create_embeddings.py` original com a função que o treino usa, [`create_dgl_graph_from_embedding`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L43).
2. Descobre que nó do Joern ocupa cada linha da matriz de features, comparando os vetores um a um.
3. Converte cada aresta do grafo DGL num par de ids do Joern e compara com as arestas do `.dot`. O `.dot` é lido com um parser próprio (`comum.ler_dot`), independente do código da Constança.
4. Repete a comparação com a correção proposta.

## Resultado

| Tipo | Arestas | Corretas (original) | Corretas (com correção) |
|---|---|---|---|
| AST | 133 | 15 | 133 |
| CFG | 63 | 14 | 63 |
| PDG | 132 | 21 | 132 |
| **Total** | **328** | **50 (15,2%)** | **328 (100%)** |

Exemplo, no CFG de `compute`:

```
aresta real (.dot)                      | aresta que a GNN recebe
unused = 5         -> a + b             | r > 10             -> helper(r)
r = a + b          -> r > 10            | r = 10             -> compute
a + b              -> r = a + b         | helper(r)          -> r = 10
```

As poucas arestas "certas" são coincidências da permutação (por exemplo, o PDG de `get_len`, com 4 nós).

## Porque acontece

São usadas duas ordens diferentes para os mesmos nós:

- [create_embeddings.py:130](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L130) numera os nós para o `edge_index` com `sorted(set(...))` sobre os ids **como texto**;
- [load_graphcodebert_embeddings.py:43](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L43) empilha as features por ordem **numérica** (`key=int`).

Os ids do Joern têm 11 dígitos (CALL, IDENTIFIER, BLOCK…) ou 12 (METHOD, METHOD_RETURN, PARAM, RETURN…). Por isso `"107374182400" < "30064771072"` como texto, mas não como número. Todos os grafos de funções têm METHOD (12 dígitos) e chamadas (11 dígitos), logo a permutação acontece em praticamente todos.

Um segundo efeito agrava o primeiro: o `edge_index` só numera os nós que aparecem em arestas, e as features incluem todos os nós do `.txt`. Um nó isolado desloca a numeração. Nos grafos de teste não há nós isolados; nos grafos reais do Linux não foi possível medir.

**Aplica-se aos dados da Constança:** o apêndice do relatório final mostra ids de 11 e 12 dígitos no mesmo grafo (`128849018880`, `30064771072`), que é a condição para a permutação.

## Impacto nos resultados da tese

Todos os resultados da DGCNN (Tabelas 5.3 e 5.8) foram obtidos com grafos em que o texto de cada nó está numa posição errada da estrutura. A GNN recebe a forma certa do grafo, mas com o conteúdo baralhado, por isso não pode aprender relações estruturais entre fragmentos de código. Isto ajuda a explicar porque é que AST, CFG e PDG dão resultados parecidos. O script [`fusion_all_graphs.py`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/fusion_all_graphs.py) faz o alinhamento corretamente, mas esse ficheiro não é usado no treino.

## Como resolver

Usar **uma só ordem de nós** para as features e para as arestas. Em `create_embeddings.py`, substituir `parse_dot_edges` por:

```python
def parse_dot_edges(dot_path, node_ids):
    """edge_index na MESMA ordem que o loader usa para as features (id numérico)."""
    ordem = sorted(node_ids, key=int)
    node_to_idx = {n: i for i, n in enumerate(ordem)}
    with open(dot_path, encoding="utf-8", errors="ignore") as f:
        edges = re.findall(r'"(\d+)"\s*->\s*"(\d+)"', f.read())
    pares = [[node_to_idx[s], node_to_idx[d]] for s, d in edges if s in node_to_idx and d in node_to_idx]
    edge_index = torch.tensor(pares, dtype=torch.long).t().reshape(2, -1)
    return edge_index, node_to_idx
```

e chamá-la com `parse_dot_edges(full_dot_path, node_embeddings.keys())` (linha 283). Esta é a ordenação usada na demonstração do teste, que recupera 328/328 arestas.

**Não é preciso recalcular as embeddings.** Os `.pt` já guardam `embeddings` e `dot_path`, por isso basta um script que, para cada `.pt`, volte a calcular `edge_index` e `node_to_idx` com esta função e grave o ficheiro. O modo `--update-edges` existente não serve, porque salta ficheiros que já têm `edge_index`.

Para evitar que volte a acontecer: o loader deve ler `node_to_idx` do `.pt` e confirmar que coincide com a ordem que usa (`assert`), e este teste deve passar a ser corrido sempre que a extração mudar.

## Limitações desta verificação

- Feita com 18 grafos de um ficheiro de teste, não com os `.dot`/`.pt` reais do Linux, que não estão nesta máquina.
- O mecanismo é determinístico e depende só do formato dos ids, por isso aplica-se aos dados reais. A percentagem exata de arestas corretas nos dados reais não foi medida.
- As embeddings são aleatórias (ver 00); o alinhamento não depende delas.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
