# 01 — As arestas ligam os nós errados


> Só 15% das arestas que a DGCNN recebe ligam os nós certos. A forma do grafo mantém-se, mas o conteúdo dos nós fica trocado entre posições. A correção recupera 100%. Em testes controlados, o erro não afetou o que depende só do conteúdo e prejudicou claramente a informação estrutural do PDG, mas não a do CFG. O efeito nos resultados da tese ainda não foi medido.

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

As poucas arestas "certas" são coincidências da permutação. A exceção é o PDG de `get_len`: os seus 4 ids têm todos 12 dígitos, por isso as duas ordens coincidem e não há troca.

## Porque acontece

São usadas duas ordens diferentes para os mesmos nós:

- [create_embeddings.py:130](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L130) numera os nós para o `edge_index` com `sorted(set(...))` sobre os ids **como texto**;
- [load_graphcodebert_embeddings.py:43](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L43) empilha as features por ordem **numérica** (`key=int`).

Os ids do Joern têm 11 dígitos (CALL, IDENTIFIER, BLOCK…) ou 12 (METHOD, METHOD_RETURN, PARAM, RETURN…). Por isso `"107374182400" < "30064771072"` como texto, mas não como número. Todos os grafos de funções têm METHOD (12 dígitos) e chamadas (11 dígitos), logo a permutação acontece em praticamente todos.

Um segundo efeito agrava o primeiro: o `edge_index` só numera os nós que aparecem em arestas, e as features incluem todos os nós do `.txt`. Um nó isolado desloca a numeração. Nos grafos de teste não há nós isolados; nos grafos reais do Linux não foi possível medir.

**Aplica-se aos dados da Constança:** o apêndice do relatório final mostra ids de 11 e 12 dígitos no mesmo grafo (`128849018880`, `30064771072`), que é a condição para a permutação.

## O que muda no grafo

Nos 18 grafos de teste:

- a forma do grafo e o conjunto de textos coincidem sempre com os do `.dot`;
- 266 de 270 nós (99%) ficam com o texto de outro nó, e 184 (68%) com o texto de um nó de outro tipo;
- a troca é sempre uma rotação: cada nó recebe o texto do nó `m` posições à frente, em que `m` é o número de ids com 12 dígitos.

Como `m` depende, por exemplo, do número de parâmetros e de `return`, o mesmo código é baralhado de forma diferente em funções diferentes. No CFG de `compute` (`m` = 3):

| Nó real | Texto que a GNN vê nesse nó | Texto que lhe chega pelas arestas | Devia chegar |
|---|---|---|---|
| `r > 10` | `compute` | `r = 10` | `r = a + b` |
| `r = 10` | `int` | `compute` | `r > 10` |
| `helper(r)` | `return helper(r);` | `compute`, `int` | `r > 10`, `r = 10` |

## Efeito na aprendizagem (experiência controlada)

[`aprofundamento.py`](aprofundamento.py) gera 600 funções C sintéticas, passa-as pelo Joern (CFG e PDG) e compara o grafo com o erro (`parse_dot_edges` e loader originais) com o grafo corrigido, em duas tarefas:

- **Conteúdo:** distinguir `memcpy` de `strncpy`.
- **Estrutura:** as instruções são as mesmas e só muda se a verificação do tamanho vem antes ou depois do `memcpy`.

**Sonda linear:** regressão logística sobre os pares de textos ligados por cada aresta. A coluna "sem arestas" usa só a contagem de textos dos nós.

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo (CFG e PDG) | 100% | 100% | 100% |
| Estrutura, CFG | 74,3% | 84,3% | 47,0% |
| Estrutura, PDG | 99,7% | 67,7% | 52,7% |
| Estrutura, CFG, com `m` diferente entre treino e teste | 68,5% | 71,2% | – |
| Estrutura, PDG, com `m` diferente entre treino e teste | 100% | 57,4% | – |

**DGCNN original:** classes e hiperparâmetros de `detect_vulnerabilities.py`, com a mediana da exatidão no teste e o mínimo e máximo de 5 seeds. Na divisão de teste, a classe minoritária tem cerca de 42%, por isso uma exatidão de 42,2% é compatível com um modelo que prevê sempre a mesma classe (não confirmado).

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo, CFG | 77,8% [49–100] | 100% [49–100] | 98,9% [86–100] |
| Conteúdo, PDG | 56,7% [49–60] | 51,1% [49–86] | 100% [51–100] |
| Estrutura, CFG | 42,2% [42–76] | 42,2% [42–52] | 42,2% [42–56] |
| Estrutura, PDG | 74,4% [42–96] | 42,2% [42–58] | 42,2% [42–48] |

O que estes resultados mostram:

- **Conteúdo:** na sonda linear, o erro não tem efeito, porque o conjunto de textos não muda.
- **PDG:** o erro reduz a informação estrutural, sobretudo em funções com assinaturas diferentes das do treino.
- **CFG:** o erro não prejudicou e deu até um valor mais alto na sonda. A causa não foi investigada. Portanto, o erro não reduz o grafo a ruído puro.
- **DGCNN:** o único sinal a favor da correção é o PDG na tarefa de estrutura, com grande variação entre seeds. O resto é inconclusivo, porque o treino é muito instável.
- **Sem arestas:** na tarefa de conteúdo, a versão sem arestas supera a corrigida, sobretudo no PDG, ou seja, corrigir o erro pode não bastar para melhorar os resultados. A causa não foi investigada.

## Impacto nos resultados da tese

Todos os resultados da DGCNN (Tabelas 5.3 e 5.8) foram obtidos com grafos em que o texto de cada nó está numa posição errada da estrutura. A GNN recebe a forma certa do grafo, mas com o conteúdo baralhado, por isso as relações que vê entre fragmentos de código não correspondem ao código. Esses padrões podem ainda conter sinal, como mostra o CFG acima.

A gravidade CRÍTICA reflete o facto de a representação não ser a que a tese descreve. A dimensão do efeito nas tabelas não foi medida, e estes resultados não se transferem diretamente para elas. A ideia de que o erro explica os resultados parecidos entre AST, CFG e PDG é apenas uma hipótese, e a AST não entrou nesta experiência. O script [`fusion_all_graphs.py`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/fusion_all_graphs.py) faz o alinhamento corretamente, mas esse ficheiro não é usado no treino.

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

Para medir o impacto real, é preciso aplicar a correção aos `.pt` do Linux, voltar a treinar e comparar com as tabelas da tese.

## Limitações desta verificação

- Feita com 18 grafos de um ficheiro de teste, não com os `.dot`/`.pt` reais do Linux, que não estão nesta máquina.
- O mecanismo é determinístico e depende só do formato dos ids, por isso aplica-se aos dados reais. A percentagem exata de arestas corretas nos dados reais não foi medida.
- As embeddings são aleatórias (ver 00); o alinhamento não depende delas.
- A rotação só foi verificada nestes 18 grafos, que não têm nós isolados. Nos dados do Linux, o padrão pode ser diferente.
- A experiência de aprendizagem usa funções sintéticas e uma substituta determinística do GraphCodeBERT (soma de vetores fixos por token), com uma única divisão treino/teste.
- A sonda linear só vê pares de nós ligados diretamente, enquanto a DGCNN agrega até 4 saltos.
- O veredicto impresso por `aprofundamento.py` é texto fixo, não é calculado a partir dos resultados.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\teste.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\aprofundamento.py
```

Precisam da preparação (00) e o segundo também do Joern. O treino da DGCNN pode demorar horas; `--sem-dgcnn` salta essa parte. As saídas estão em [resultado.txt](resultado.txt) e [resultado_aprofundamento.txt](resultado_aprofundamento.txt).
