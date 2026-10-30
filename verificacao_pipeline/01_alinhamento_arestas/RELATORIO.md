# 01 — As arestas ligam os nós errados

## Como está organizada esta pasta

Cada linha responde a uma pergunta, pela ordem em que convém explicar:

| Pergunta | Resposta curta | Script | Saída | Secção |
|---|---|---|---|---|
| 1. O erro existe? | Sim: só 15% das arestas estão certas e a correção recupera 100% | [teste.py](teste.py) | [resultado.txt](resultado.txt) | [Resultado](#resultado) |
| 2. O que muda no grafo? | A forma mantém-se, mas os textos trocam de nó | [teste_aprofundado.py](aprendizagem/teste_aprofundado.py) (Partes A e B) | [resultado_testes_aprofundados.txt](aprendizagem/resultado_testes_aprofundados.txt) | [O que muda no grafo](#o-que-muda-no-grafo) |
| 3. Também acontece em código real? | Sim: 7 a 12% de arestas certas em 135 funções do kernel | [kernel_real.py](kernel_real/kernel_real.py) | [resultado_kernel_real.txt](kernel_real/resultado_kernel_real.txt) | [Confirmação em código real](#confirmação-em-código-real-do-kernel) |
| 4. Afeta o que o modelo aprende? | No PDG, sim; na DGCNN, o resultado é pouco conclusivo porque o treino é instável | [teste_aprofundado.py](aprendizagem/teste_aprofundado.py) (Parte C) | [resultado_testes_aprofundados.txt](aprendizagem/resultado_testes_aprofundados.txt) | [Efeito na aprendizagem](#efeito-na-aprendizagem-experiência-controlada) |
| 5. Porque é que a DGCNN falha tantas vezes? | Colapsa numa só classe; a causa não está provada | [diagnostico_sintetico.py](aprendizagem/diagnostico_sintetico.py) | [resultado_diagnostico_sintetico.txt](aprendizagem/resultado_diagnostico_sintetico.txt) | [Efeito na aprendizagem](#efeito-na-aprendizagem-experiência-controlada) |

A pasta [falhas_loader/](falhas_loader/) trata de outro problema (grafos que não carregam no loader). Está em stand-by para já.

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

Um segundo efeito agrava o primeiro: o `edge_index` só numera os nós que aparecem em arestas, e as features incluem todos os nós do `.txt`. Um nó isolado desloca a numeração. Nos grafos de teste não há nós isolados. Em código real do kernel, há em 9 de 133 PDG e em nenhum grafo AST ou CFG, e nesses 9 PDG a troca deixa de ser uma rotação exata.

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

## Confirmação em código real do kernel

[`kernel_real.py`](kernel_real/kernel_real.py) corre a pipeline original (Joern, `generate_txt_force`, `parse_dot_edges` e loader) sobre os 4 ficheiros de [dados/linux_src/](../dados/linux_src/): `ioremap`, `ip_sockglue` e `sctp_socket` em 2 commits. São 135 funções e 405 grafos. Mede o alinhamento, por isso não depende do valor das embeddings nem usa treino.

| Grafo | Grafos medidos | Arestas certas | Nós trocados | Nós com texto de outro tipo | Rotação exata | Valores de `m` |
|---|---|---|---|---|---|---|
| AST | 99 | 12,0% | 100% | 43,1% | 99 de 99 | 3 a 13 |
| CFG | 135 | 12,1% | 99,8% | 19,8% | 135 de 135 | 2 a 11 |
| PDG | 133 | 7,1% | 99,7% | 33,0% | 124 de 133 | 3 a 12 |

- O mecanismo confirma-se em código real, e o `m` varia de função para função.
- Os grafos reais são maiores do que os sintéticos usados abaixo: a mediana é de 67 nós na AST, 39 no CFG e 34 no PDG, e 53 a 76% têm mais de 32 nós, o `k` do SortPooling.
- Os 38 grafos que faltam (36 AST e 2 PDG) não carregam no loader e ficam fora destas contagens. Este efeito é separado da troca de ordem e não é tratado neste relatório.

## Efeito na aprendizagem (experiência controlada)

[`teste_aprofundado.py`](aprendizagem/teste_aprofundado.py) gera 600 funções C sintéticas, passa-as pelo Joern (CFG e PDG) e compara o grafo com o erro (`parse_dot_edges` e loader originais) com o grafo corrigido, em duas tarefas:

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

**DGCNN original:** classes e hiperparâmetros de `detect_vulnerabilities.py`, com a mediana da exatidão no teste e o mínimo e máximo de 5 seeds. Na tarefa de estrutura, a classe mais frequente no treino é a minoritária no teste (cerca de 42%). Por isso, 42,2% corresponde a um modelo que prevê sempre a mesma classe, o que [`diagnostico_sintetico.py`](aprendizagem/diagnostico_sintetico.py) confirmou na seed 0.

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo, CFG | 77,8% [49–100] | 100% [49–100] | 98,9% [86–100] |
| Conteúdo, PDG | 56,7% [49–60] | 51,1% [49–86] | 100% [51–100] |
| Estrutura, CFG | 42,2% [42–76] | 42,2% [42–52] | 42,2% [42–56] |
| Estrutura, PDG | 74,4% [42–96] | 42,2% [42–58] | 42,2% [42–48] |

O que estes resultados mostram:

- **Conteúdo:** na sonda linear, o erro não tem efeito, porque o conjunto de textos não muda.
- **PDG:** o erro reduz a informação estrutural, sobretudo em funções com assinaturas diferentes das do treino. A tabela seguinte mostra em quantos grafos, de 150 por classe, a GNN vê uma aresta verificação → `memcpy`. No grafo corrigido, esta aresta quase separa as classes sozinha; com o erro, o sinal desaparece.

  | Grafo | Com a verificação antes | Com o `memcpy` antes |
  |---|---|---|
  | PDG corrigido | 147 | 0 |
  | PDG com o erro | 15 | 17 |

- **CFG:** o erro não prejudicou na sonda, mas isso não é uma vantagem. O grafo corrigido fica limitado porque a sonda só vê vizinhos diretos, e a verificação e o `memcpy` só são adjacentes em cerca de um terço dos casos. Com o erro, a sonda encontra um sinal de sentido invertido: a aresta verificação → `memcpy` aparece em 32 grafos com o `memcpy` antes e em 2 com a verificação antes, e o seu peso passa de −2,15 para +2,16. O modelo acerta por um artefacto dos ids, não pela relação real no código. Não foi testado se uma sonda com mais saltos ultrapassaria os 84%.
- **DGCNN:** os casos com 42,2% não comparam o erro, porque o treino falha. Na seed 0, nesses casos, a exatidão no treino fica em 53,3%, a perda em 0,69 (ln 2) e as previsões numa só classe, com o grafo corrigido e com o erro. A causa do colapso não foi provada. O melhor indício é o PDG na tarefa de estrutura: com o grafo corrigido, pelo menos 3 de 5 seeds chegam a 74% ou mais; com o erro, nenhuma passa de 58%. A amostra é pequena e a tarefa foi desenhada para este efeito.
- **Sem arestas:** na tarefa de conteúdo, a versão sem arestas supera a corrigida, sobretudo no PDG. Na seed 0 desse PDG, o grafo corrigido colapsa numa só classe, enquanto a versão sem arestas chega a 100%. Portanto, corrigir o erro pode não bastar sem estabilizar o treino. Não se sabe porque é que as arestas pioram o treino.

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

- Os `.dot`/`.pt` do dataset real do Linux não estão nesta máquina. O alinhamento foi medido em 18 grafos de um ficheiro de teste e em 135 funções reais do kernel, de 3 ficheiros distintos, um deles em 2 commits. Esta amostra não representa o dataset, por isso as percentagens são indicativas.
- O mecanismo é determinístico e depende só do formato dos ids, por isso aplica-se aos dados reais.
- As embeddings dos grafos de teste são aleatórias (ver 00); o alinhamento não depende delas.
- A forma do grafo só foi comparada, aresta a aresta, nos 18 grafos de teste. A correção também só foi aplicada a esses grafos e aos sintéticos, não ao código real do kernel.
- A experiência de aprendizagem usa funções sintéticas simples, com classes equilibradas (a tese tem cerca de 2,6% de vulneráveis) e uma substituta determinística do GraphCodeBERT (soma de vetores fixos por token), com uma única divisão treino/teste. As funções são mais pequenas do que as reais (16 a 24 nós de mediana) e quase nunca são cortadas pelo SortPooling. A AST não entrou nesta experiência.
- O diagnóstico do colapso da DGCNN usa só a seed 0.
- A sonda linear só vê pares de nós ligados diretamente, enquanto a DGCNN agrega até 4 saltos.
- O veredicto impresso por `teste_aprofundado.py` é texto fixo, não é calculado a partir dos resultados.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\teste.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\aprendizagem\teste_aprofundado.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\kernel_real\kernel_real.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\aprendizagem\diagnostico_sintetico.py
```

O `teste.py` e o `teste_aprofundado.py` precisam da preparação (00); os três últimos precisam do Joern. O treino da DGCNN pode demorar horas; no `teste_aprofundado.py`, `--sem-dgcnn` salta essa parte. O `kernel_real.py` demora alguns minutos e o `diagnostico_sintetico.py` cerca de meia hora.

Os comandos correm a partir de `verificacao_pipeline`. As saídas estão indicadas na [tabela do início](#como-está-organizada-esta-pasta).
