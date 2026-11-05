# 01 — Os textos dos nós ficam no sítio errado do grafo

Este relatório explica:

1. qual é o erro;
2. o que muda no grafo;
3. que dano causa (na representação, na aprendizagem e nos resultados);
4. quão crítico é;
5. como se corrige.

No fim estão os testes feitos, as limitações e como correr tudo.

## 1. O erro

A pipeline guarda os nós de cada grafo em **duas ordens diferentes**:

- [create_embeddings.py:130](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L130) numera os nós para o `edge_index` (as arestas) com `sorted(set(...))` sobre os ids **como texto**.
- [load_graphcodebert_embeddings.py:43](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L43) empilha as features (os vetores GraphCodeBERT de cada nó) por ordem **numérica** (`key=int`).

A aresta "nó 3 → nó 7" refere-se, portanto, a nós diferentes nas arestas e nas features.

**Porque é que as duas ordens diferem.** Os ids do Joern têm 11 ou 12 dígitos:

- 11 dígitos: CALL, IDENTIFIER, BLOCK, …
- 12 dígitos: METHOD, METHOD_RETURN, PARAM, RETURN, …

Como texto, `"107374182400" < "30064771072"`; como número, não. Todas as funções têm um METHOD (12 dígitos) e chamadas (11 dígitos), por isso a troca acontece em praticamente todos os grafos. O apêndice do relatório da Constança mostra ids dos dois tamanhos no mesmo grafo (`128849018880`, `30064771072`), que é a condição para a troca.

**Um segundo efeito agrava o primeiro.** O `edge_index` só numera os nós que aparecem em arestas, mas as features incluem todos os nós do `.txt`. Um nó isolado desloca a numeração. Em código real isto só acontece em alguns PDG (17 de 2007 na amostra real).

## 2. O que muda no grafo

**A forma do grafo não muda. Mudam os textos que estão em cada nó.**

- Sem nós isolados, a troca é uma rotação: cada nó recebe o texto do nó `m` posições à frente, em que `m` é o número de ids com 12 dígitos.
- O número de nós e de arestas e quem liga a quem mantêm-se. O que muda é qual fragmento de código está em cada posição.
- `m` depende, por exemplo, do número de parâmetros e de `return`, por isso o mesmo código é baralhado de forma diferente em funções diferentes.

Exemplo, no CFG de `compute` (`m` = 3):

| Nó real | Texto que a GNN vê nesse nó | Texto que lhe chega pelas arestas | Devia chegar |
|---|---|---|---|
| `r > 10` | `compute` | `r = 10` | `r = a + b` |
| `r = 10` | `int` | `compute` | `r > 10` |
| `helper(r)` | `return helper(r);` | `compute`, `int` | `r > 10`, `r = 10` |

Na amostra real do kernel, com 5984 grafos (ver [secção 6](#6-como-foi-testado)):

| Grafo | Estrutura igual à do `.dot` | Nós com o texto de outro nó | Arestas que ligam o par de textos certo |
|---|---|---|---|
| AST | 1796 de 1796 | 98% | 14,8% |
| CFG | 2181 de 2181 | 97% | 14,1% |
| PDG | 2005 de 2007 | 96% | 8,8% |

Os 2 PDG com estrutura diferente têm um nó sem texto no `.txt`. As poucas arestas "certas" são coincidências da rotação ou de textos repetidos.

## 3. O dano

### 3.1 Na representação

A tese descreve uma GNN que aprende relações entre instruções: quem controla quem (CFG), quem depende de quem (PDG) e quem contém quem (AST). Com o erro, a GNN recebe a forma certa, mas 85 a 91% das relações que vê ligam fragmentos de código que não estão ligados no programa. **A representação usada não é a que a tese descreve**, em todos os tipos de grafo e em todos os resultados GraphCodeBERT + DGCNN (Tabelas 5.4 a 5.8). A Tabela 5.3 usa features manuais e não passa por este código.

O que **não** se perde: o conjunto de textos de cada grafo é o mesmo. Um modelo que só olhe para "que instruções existem" (por exemplo, se há um `memcpy`) não é afetado.

### 3.2 Na aprendizagem (experiência controlada)

[`teste_aprofundado.py`](aprendizagem/teste_aprofundado.py) gera 600 funções C sintéticas e passa-as pelo Joern (CFG e PDG). Depois compara o grafo com o erro (`parse_dot_edges` e loader originais) com o grafo corrigido, em duas tarefas:

- **Conteúdo:** distinguir `memcpy` de `strncpy`.
- **Estrutura:** as instruções são as mesmas e só muda se a verificação do tamanho vem antes ou depois do `memcpy`.

**Sonda linear.** É uma regressão logística sobre os pares de textos ligados por cada aresta. A coluna "sem arestas" usa só a contagem de textos dos nós.

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo (CFG e PDG) | 100% | 100% | 100% |
| Estrutura, CFG | 74,3% | 84,3% | 47,0% |
| Estrutura, PDG | 99,7% | 67,7% | 52,7% |
| Estrutura, CFG, com `m` diferente entre treino e teste | 68,5% | 71,2% | – |
| Estrutura, PDG, com `m` diferente entre treino e teste | 100% | 57,4% | – |

- **Conteúdo:** o erro não tem efeito, como esperado.
- **PDG:** o erro apaga o sinal estrutural. A aresta verificação → `memcpy` aparece da seguinte forma, em 150 grafos por classe:

  | Grafo | Com a verificação antes | Com o `memcpy` antes |
  |---|---|---|
  | PDG corrigido | 147 | 0 |
  | PDG com o erro | 15 | 17 |

  No grafo corrigido, esta aresta quase separa as classes sozinha; com o erro, deixa de separar. Com o erro, o resultado também cai quando `m` muda entre treino e teste (57%), ou seja, quando as assinaturas das funções diferem.
- **CFG:** o erro não piorou a sonda, mas por um artefacto. A sonda encontra um sinal de sentido invertido: a aresta verificação → `memcpy` aparece em 32 grafos com o `memcpy` antes e em 2 com a verificação antes, e o peso passa de −2,15 para +2,16. O modelo acerta por causa dos ids, não pela relação real no código. O grafo corrigido fica limitado porque a sonda só vê vizinhos diretos, e a verificação e o `memcpy` só são adjacentes em cerca de um terço dos casos.

**DGCNN original.** Usa as classes e os hiperparâmetros de `detect_vulnerabilities.py`. A tabela mostra a mediana da exatidão no teste e, entre parênteses retos, o mínimo e o máximo de 5 seeds. Na tarefa de estrutura, 42,2% corresponde a prever sempre a mesma classe.

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo, CFG | 77,8% [49–100] | 100% [49–100] | 98,9% [86–100] |
| Conteúdo, PDG | 56,7% [49–60] | 51,1% [49–86] | 100% [51–100] |
| Estrutura, CFG | 42,2% [42–76] | 42,2% [42–52] | 42,2% [42–56] |
| Estrutura, PDG | 74,4% [42–96] | 42,2% [42–58] | 42,2% [42–48] |

- **PDG na tarefa de estrutura:** é o melhor indício de dano. Com o grafo corrigido, pelo menos 3 de 5 seeds chegam a 74% ou mais; com o erro, nenhuma passa de 58%.
- **Colapsos:** o treino colapsa muitas vezes numa só classe. [`diagnostico_sintetico.py`](aprendizagem/diagnostico_sintetico.py) mostrou, na seed 0, perda de 0,69 (ln 2) e exatidão de 53% no treino, com o erro e corrigido. A causa do colapso não foi provada e não vem deste erro.
- **Sem arestas:** na tarefa de conteúdo, a versão sem arestas supera a corrigida. Corrigir o erro pode não bastar sem estabilizar o treino.

### 3.3 Nos resultados (treino numa amostra real)

Reproduzi o caminho da Tabela 5.7, que usa as classes e os hiperparâmetros de `detect_vulnerabilities.py`:

- DGCNN T9 com 30 épocas;
- embeddings do encoder;
- SMOTETomek;
- SVM ou XGBoost com parâmetros por omissão.

O treino correu sobre 5984 grafos reais com o GraphCodeBERT real, em três condições que só diferem nas arestas:

- **com o erro:** a pipeline original;
- **corrigido:** uma só ordem de nós;
- **sem arestas.**

Os grafos, as features, a divisão treino/teste e a inicialização são os mesmos nas três condições. Os resultados são F1 e PR-AUC no teste, em média de 3 seeds (42, 53 e 64):

| Classificador | Grafo | F1 com o erro | F1 corrigido | F1 sem arestas | PR-AUC com o erro | PR-AUC corrigido | PR-AUC sem arestas |
|---|---|---|---|---|---|---|---|
| SVM | AST | 0,220 | 0,231 | 0,238 | 0,161 | 0,149 | 0,161 |
| SVM | CFG | 0,266 | 0,276 | 0,294 | 0,167 | 0,185 | 0,203 |
| SVM | PDG | 0,256 | 0,247 | 0,271 | 0,173 | 0,174 | 0,232 |
| XGBoost | AST | 0,130 | 0,182 | 0,227 | 0,143 | 0,212 | 0,228 |
| XGBoost | CFG | 0,159 | 0,229 | 0,233 | 0,178 | 0,201 | 0,220 |
| XGBoost | PDG | 0,140 | 0,240 | 0,226 | 0,159 | 0,233 | 0,226 |

- **SVM:** com o erro e corrigido dão o mesmo. As diferenças por seed têm sinais trocados e são menores do que a variação entre seeds.
- **XGBoost:** o erro custa cerca de 0,05 a 0,10 de F1. No PDG, a correção melhora nas 3 seeds (F1 +0,06 a +0,15; PR-AUC +0,05 a +0,10); no CFG, em 2 de 3.
- **Sem arestas:** fica sempre ao nível do corrigido ou acima. A correção retira o ruído dos textos trocados, mas não faz a estrutura acrescentar informação. Nesta configuração, a DGCNN não tira proveito das arestas, com ou sem o erro.
- **Conv1D, o classificador da própria DGCNN:** em 11 dos 27 treinos prevê uma só classe no teste, por isso não permite comparar condições.
- **Nível absoluto:** recalculado para a prevalência da tese (2,6% de vulneráveis), o F1 fica entre 0,07 e 0,13 em todas as condições.

Detalhe e restantes métricas em [resultado_impacto.txt](impacto_tese/resultado_impacto.txt).

## 4. Quão crítico é

A gravidade depende da pergunta. Por isso é avaliada em três planos.

| Plano | Gravidade | Porquê |
|---|---|---|
| Correção da representação | **Crítica** | Em praticamente todos os grafos, 85 a 91% das arestas ligam os textos errados. O modelo treinado não é o que a tese descreve. |
| Valor das métricas (Tabelas 5.4–5.8) | **Moderada** | Na amostra real: nenhum efeito visível no SVM; cerca de 0,05 a 0,10 de F1 a menos no XGBoost. Os números da tese provavelmente estão um pouco abaixo do que dariam com a correção, mas não mudam de ordem de grandeza. |
| Conclusões sobre a estrutura (AST vs CFG vs PDG, papel da GNN) | **Alta** | Com o erro, nenhuma comparação entre tipos de grafo mede de facto a estrutura. Mesmo corrigido, o modelo sem arestas fica ao mesmo nível. Logo, a tese não demonstra que a estrutura ajuda, e isso não se resolve só com esta correção. |

Em resumo:

- O erro é real, determinístico e afeta todos os resultados GraphCodeBERT + DGCNN.
- O seu peso nas métricas é limitado, porque o modelo, neste caminho, se apoia sobretudo no conteúdo dos nós, que o erro não altera.
- O problema mais sério para as conclusões é que a estrutura não contribui, com ou sem o erro.

## 5. Como resolver

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

e chamá-la com `parse_dot_edges(full_dot_path, node_embeddings.keys())` (linha 283). Também resolve os nós isolados, porque a numeração passa a incluir todos os nós. No ficheiro de teste recupera 328 de 328 arestas.

- **Não é preciso recalcular as embeddings.** Os `.pt` já guardam `embeddings` e `dot_path`. Basta um script que, para cada `.pt`, volte a calcular `edge_index` e `node_to_idx` com esta função e grave o ficheiro. O modo `--update-edges` existente não serve, porque salta ficheiros que já têm `edge_index`.
- **Para não voltar a acontecer:** o loader deve ler `node_to_idx` do `.pt` e confirmar com `assert` que coincide com a ordem que usa. Este teste deve ser corrido sempre que a extração mudar.
- **Para medir o impacto exato nas tabelas:** aplicar a correção aos `.pt` do Linux completos e voltar a treinar.
- **Para saber se a estrutura ajuda:** comparar sempre com a versão sem arestas, como na [secção 3.3](#33-nos-resultados-treino-numa-amostra-real).

O script [`fusion_all_graphs.py`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/fusion_all_graphs.py) faz o alinhamento corretamente, mas não é usado no treino.

## 6. Como foi testado

Os testes vão do mais simples ao mais próximo da tese:

| Teste | Que código | O que mede | Script | Saída |
|---|---|---|---|---|
| Existência do erro | 18 grafos de um ficheiro C de teste | Arestas certas com o erro (15%) e com a correção (100%) | [teste.py](teste.py) | [resultado.txt](resultado.txt) |
| Forma do grafo e aprendizagem | 600 funções C sintéticas, feitas para isolar conteúdo e estrutura | Rotação dos textos; sonda linear e DGCNN com o erro, corrigido e sem arestas | [teste_aprofundado.py](aprendizagem/teste_aprofundado.py) | [resultado_testes_aprofundados.txt](aprendizagem/resultado_testes_aprofundados.txt) |
| Colapso da DGCNN | As mesmas funções sintéticas, seed 0 | Perda, exatidão no treino e previsões | [diagnostico_sintetico.py](aprendizagem/diagnostico_sintetico.py) | [resultado_diagnostico_sintetico.txt](aprendizagem/resultado_diagnostico_sintetico.txt) |
| Código real (pequeno) | 135 funções de 4 ficheiros do kernel (`ioremap`, `ip_sockglue`, `sctp_socket` em 2 commits) | Arestas certas (7 a 12%) e rotação exata | [kernel_real.py](kernel_real/kernel_real.py) | [resultado_kernel_real.txt](kernel_real/resultado_kernel_real.txt) |
| Estrutura em código real | 5984 grafos de 160 ficheiros do kernel sorteados do dataset da tese | Se a estrutura é igual à do `.dot` e quantos textos trocam | [verificar_estrutura.py](impacto_tese/verificar_estrutura.py) | [resultado_verificar_estrutura.txt](impacto_tese/resultado_verificar_estrutura.txt) |
| Impacto nos resultados | Os mesmos 5984 grafos, GraphCodeBERT real | Caminho da Tabela 5.7, nas três condições e com 3 seeds | [treinar.py](impacto_tese/treinar.py), [resumir.py](impacto_tese/resumir.py) | [resultado_impacto.txt](impacto_tese/resultado_impacto.txt) |

**Como se verifica o alinhamento.** Cada grafo é carregado com o loader original, [`create_dgl_graph_from_embedding`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L43). Descobre-se que nó do Joern ocupa cada linha da matriz de features e converte-se cada aresta num par de ids. Esse par é comparado com as arestas do `.dot`, lidas com um parser próprio (`comum.ler_dot`), independente do código da Constança.

**Amostra real.** [`construir_amostra.py`](impacto_tese/construir_amostra.py) sorteia pares (commit, ficheiro) dos CSV reais com pelo menos uma função vulnerável. De cada um guarda todas as vulneráveis e até 15 não vulneráveis. Depois corre a extração original: Joern, `generate_txt_force` e `parse_dot_edges`. [`gerar_embeddings.py`](impacto_tese/gerar_embeddings.py) calcula as embeddings com `microsoft/graphcodebert-base`. Só entram os grafos que o loader carrega, como na tese:

| Grafo | Grafos | Vulneráveis | Perdidos no loader |
|---|---|---|---|
| AST | 1796 | 157 | 432 |
| CFG | 2181 | 234 | 22 |
| PDG | 2007 | 204 | 221 |

A perda de grafos no loader é outro problema, tratado em [falhas_loader/](falhas_loader/), que está em stand-by para já.

## 7. Limitações

- **Dataset completo:** os `.pt` do Linux da tese não estão nesta máquina. A amostra real foi reconstruída com a extração original, mas é uma amostra:
  - 160 ficheiros;
  - cerca de 10% de vulneráveis, contra 2,6% na tese;
  - por isso, as métricas absolutas não são comparáveis às tabelas.
- **Treino na amostra real:**
  - 3 seeds em vez de 5;
  - só CPU;
  - sem afinação com Optuna (Tabela 5.8);
  - sem o caminho AMP/VGG.
- **Experiência sintética:**
  - as funções são simples e mais pequenas do que as reais;
  - as classes são equilibradas;
  - o GraphCodeBERT é substituído por uma soma de vetores fixos por token;
  - não inclui a AST;
  - há uma única divisão treino/teste.
- **Diagnóstico do colapso:** usa só a seed 0.
- **Sonda linear:** só vê pares de nós ligados diretamente, enquanto a DGCNN agrega até 4 saltos.
- **Veredicto de `teste_aprofundado.py`:** o que o script imprime é texto fixo, não calculado a partir dos resultados.
- **Mecanismo:** é determinístico e depende só do formato dos ids, por isso aplica-se ao dataset completo, mesmo sem o ter.

## 8. Como correr

A partir de `verificacao_pipeline`:

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\teste.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\aprendizagem\teste_aprofundado.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\kernel_real\kernel_real.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\aprendizagem\diagnostico_sintetico.py
```

- O `teste.py` e o `teste_aprofundado.py` precisam da preparação (00).
- Todos, exceto o `teste.py`, precisam do Joern.
- O `teste_aprofundado.py` pode demorar horas por causa da DGCNN; `--sem-dgcnn` salta essa parte.
- O `kernel_real.py` demora alguns minutos e o `diagnostico_sintetico.py` cerca de meia hora.

Amostra real, por esta ordem, em `01_alinhamento_arestas\impacto_tese\`:

1. `construir_amostra.py --n-ficheiros 160 --workers 2` (cerca de 2 h);
2. `gerar_embeddings.py` (cerca de 50 min);
3. `verificar_estrutura.py`;
4. `correr_treinos.py 2` (cerca de 6 h para os 27 treinos);
5. `resumir.py`.
