# 01 — Material de apoio

Pormenores que sustentam o [RELATORIO.md](RELATORIO.md): todas as tabelas das experiências, como foram feitas, as limitações e como correr tudo. O que mudou no código e os testes da correção estão em [correcao/README.md](correcao/README.md).

## 1. O mecanismo em pormenor

- [create_embeddings.py:168](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L168) numera os nós do `edge_index` com `sorted(set(...))` sobre os ids **como texto**.
- [load_graphcodebert_embeddings.py:118](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L118) empilha as features por ordem **numérica** (`key=int`).

Os ids do Joern têm 11 dígitos (CALL, IDENTIFIER, BLOCK, …) ou 12 (METHOD, METHOD_RETURN, PARAM, RETURN, …). Como texto, `"107374182400" < "30064771072"`. Todas as funções têm um METHOD e chamadas, por isso a troca acontece em praticamente todos os grafos. O apêndice do relatório da Constança mostra ids dos dois tamanhos no mesmo grafo (`128849018880`, `30064771072`).

- **Rotação.** Sem nós isolados, cada nó recebe o texto do nó `m` posições à frente, em que `m` é o número de ids com 12 dígitos. `m` depende, por exemplo, do número de parâmetros e de `return`, por isso o mesmo código é baralhado de forma diferente em funções diferentes.
- **Nós isolados.** O `edge_index` só numera os nós que aparecem em arestas, mas as features incluem todos os nós do `.txt`. Um nó isolado desloca a numeração. Em código real só acontece em alguns PDG (17 de 2007 na amostra).
- **As poucas arestas "certas"** (9 a 15%) são coincidências da rotação ou de textos repetidos.
- **Os 2 PDG com estrutura diferente** da do `.dot` têm uma aresta para um nó sem texto no `.txt`.

## 2. Experiência controlada (funções sintéticas)

[`teste_aprofundado.py`](testes_erro/testes_mais_aprofundados/teste_aprofundado.py) gera 600 funções C sintéticas e passa-as pelo Joern (CFG e PDG). Depois compara o grafo com o erro (`parse_dot_edges` e loader originais) com o grafo corrigido, em duas tarefas:

- **Conteúdo:** distinguir `memcpy` de `strncpy`.
- **Estrutura:** as instruções são as mesmas e só muda se a verificação do tamanho vem antes ou depois do `memcpy`.

### 2.1 Regressão 

É uma regressão logística sobre os pares de textos ligados por cada aresta. A coluna "sem arestas" usa só a contagem de textos dos nós.

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

  Com o erro, o resultado também cai quando `m` muda entre treino e teste (57%), ou seja, quando as assinaturas das funções diferem.
- **CFG:** o erro não piorou a sonda, mas por um artefacto. A sonda encontra um sinal de sentido invertido: a aresta verificação → `memcpy` aparece em 32 grafos com o `memcpy` antes e em 2 com a verificação antes, e o peso passa de −2,15 para +2,16. O modelo acerta por causa dos ids, não pela relação real no código. O grafo corrigido fica limitado porque a sonda só vê vizinhos diretos, e a verificação e o `memcpy` só são adjacentes em cerca de um terço dos casos.

### 2.2 DGCNN original

Usa as classes e os hiperparâmetros de `detect_vulnerabilities.py`. Mediana da exatidão no teste e, entre parênteses retos, o mínimo e o máximo de 5 seeds. Na tarefa de estrutura, 42,2% corresponde a prever sempre a mesma classe.

| Teste | Corrigido | Com o erro | Sem arestas |
|---|---|---|---|
| Conteúdo, CFG | 77,8% [49–100] | 100% [49–100] | 98,9% [86–100] |
| Conteúdo, PDG | 56,7% [49–60] | 51,1% [49–86] | 100% [51–100] |
| Estrutura, CFG | 42,2% [42–76] | 42,2% [42–52] | 42,2% [42–56] |
| Estrutura, PDG | 74,4% [42–96] | 42,2% [42–58] | 42,2% [42–48] |

- **PDG na tarefa de estrutura:** é o melhor indício de dano. Com o grafo corrigido, pelo menos 3 de 5 seeds chegam a 74% ou mais; com o erro, nenhuma passa de 58%.
- **Colapsos:** o treino colapsa muitas vezes numa só classe. [`diagnostico_sintetico.py`](testes_erro/testes_mais_aprofundados/diagnostico_sintetico.py) mostrou, na seed 0, perda de 0,69 (ln 2) e exatidão de 53% no treino, com o erro e corrigido. A causa do colapso não foi provada e não vem deste erro.
- **Sem arestas:** na tarefa de conteúdo, a versão sem arestas supera a corrigida. Corrigir o erro pode não bastar sem estabilizar o treino.

## 3. Treino numa amostra real (caminho da Tabela 5.7)

Configuração, com as classes e os hiperparâmetros de `detect_vulnerabilities.py`:

- DGCNN T9 com 30 épocas;
- embeddings do encoder;
- SMOTETomek;
- SVM ou XGBoost com parâmetros por omissão.

O treino correu sobre 5984 grafos reais com o GraphCodeBERT real, em três condições que só diferem nas arestas: **com o erro** (a pipeline original), **corrigido** (uma só ordem de nós) e **sem arestas**. Os grafos, as features, a divisão treino/teste e a inicialização são os mesmos nas três. F1 e PR-AUC no teste, em média de 3 seeds (42, 53 e 64):

| Classificador | Grafo | F1 com o erro | F1 corrigido | F1 sem arestas | PR-AUC com o erro | PR-AUC corrigido | PR-AUC sem arestas |
|---|---|---|---|---|---|---|---|
| SVM | AST | 0,220 | 0,231 | 0,238 | 0,161 | 0,149 | 0,161 |
| SVM | CFG | 0,266 | 0,276 | 0,294 | 0,167 | 0,185 | 0,203 |
| SVM | PDG | 0,256 | 0,247 | 0,271 | 0,173 | 0,174 | 0,232 |
| XGBoost | AST | 0,130 | 0,182 | 0,227 | 0,143 | 0,212 | 0,228 |
| XGBoost | CFG | 0,159 | 0,229 | 0,233 | 0,178 | 0,201 | 0,220 |
| XGBoost | PDG | 0,140 | 0,240 | 0,226 | 0,159 | 0,233 | 0,226 |

- **SVM:** com o erro e corrigido dão o mesmo. As diferenças por seed têm sinais trocados e são menores do que a variação entre seeds.
- **XGBoost:** no PDG, a correção melhora nas 3 seeds (F1 +0,06 a +0,15; PR-AUC +0,05 a +0,10); no CFG, em 2 de 3.
- **Sem arestas:** fica sempre ao nível do corrigido ou acima.
- **Conv1D, o classificador da própria DGCNN:** em 11 dos 27 treinos prevê uma só classe no teste, por isso não permite comparar condições.
- **Nível absoluto:** recalculado para a prevalência da tese (2,6% de vulneráveis), o F1 fica entre 0,07 e 0,13 em todas as condições.
- **PDG:** a correção aplicada descarta 2 grafos que esta condição "corrigido" incluía (ver T6 em [correcao/README.md](correcao/README.md)). Para AST e CFG os grafos são idênticos, bit a bit, aos da correção real.

Restantes métricas e valores por seed em [resultado_impacto.txt](testes_erro/impacto_tese/resultado_impacto.txt).

## 4. Como foi testado

Os testes vão do mais simples ao mais próximo da tese:

| Teste | Que código | O que mede | Script | Saída |
|---|---|---|---|---|
| Existência do erro | 18 grafos de um ficheiro C de teste | Arestas certas com o erro (15%) e com a correção (100%) | [teste_inicial.py](teste_inicial.py) | [resultado.txt](resultado.txt) |
| Forma do grafo e aprendizagem | 600 funções C sintéticas, feitas para isolar conteúdo e estrutura | Rotação dos textos; sonda linear e DGCNN com o erro, corrigido e sem arestas | [teste_aprofundado.py](testes_erro/testes_mais_aprofundados/teste_aprofundado.py) | [resultado_testes_aprofundados.txt](testes_erro/testes_mais_aprofundados/resultado_testes_aprofundados.txt) |
| Colapso da DGCNN | As mesmas funções sintéticas, seed 0 | Perda, exatidão no treino e previsões | [diagnostico_sintetico.py](testes_erro/testes_mais_aprofundados/diagnostico_sintetico.py) | [resultado_diagnostico_sintetico.txt](testes_erro/testes_mais_aprofundados/resultado_diagnostico_sintetico.txt) |
| Código real (pequeno) | 135 funções de 4 ficheiros do kernel (`ioremap`, `ip_sockglue`, `sctp_socket` em 2 commits) | Arestas certas (7 a 12%) e rotação exata | [kernel_real.py](testes_erro/kernel_real/kernel_real.py) | [resultado_kernel_real.txt](testes_erro/kernel_real/resultado_kernel_real.txt) |
| Estrutura em código real | 5984 grafos de 160 ficheiros do kernel sorteados do dataset da tese | Se a estrutura é igual à do `.dot` e quantos textos trocam | [verificar_estrutura.py](testes_erro/impacto_tese/verificar_estrutura.py) | [resultado_verificar_estrutura.txt](testes_erro/impacto_tese/resultado_verificar_estrutura.txt) |
| Impacto nos resultados | Os mesmos 5984 grafos, GraphCodeBERT real | Caminho da Tabela 5.7, nas três condições e com 3 seeds | [treinar.py](testes_erro/impacto_tese/treinar.py), [resumir.py](testes_erro/impacto_tese/resumir.py) | [resultado_impacto.txt](testes_erro/impacto_tese/resultado_impacto.txt) |
| Correção | Os mesmos grafos (sintéticos, ficheiro de teste, amostra real) | Código corrigido vs original | [testar_correcao.py](correcao/testar_correcao.py) | [resultado_correcao.txt](correcao/resultado_correcao.txt) |

**Como se verifica o alinhamento.** Cada grafo é carregado com o loader original, [`create_dgl_graph_from_embedding`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L118). Descobre-se que nó do Joern ocupa cada linha da matriz de features e converte-se cada aresta num par de ids. Esse par é comparado com as arestas do `.dot`, lidas com um parser próprio (`comum.ler_dot`), independente do código da Constança.

**Amostra real.** [`construir_amostra.py`](testes_erro/impacto_tese/construir_amostra.py) sorteia pares (commit, ficheiro) dos CSV reais com pelo menos uma função vulnerável. De cada um guarda todas as vulneráveis e até 15 não vulneráveis. Depois corre a extração original: Joern, `generate_txt_force` e `parse_dot_edges`. [`gerar_embeddings.py`](testes_erro/impacto_tese/gerar_embeddings.py) calcula as embeddings com `microsoft/graphcodebert-base`. Só entram os grafos que o loader carrega, como na tese:

| Grafo | Grafos | Vulneráveis | Perdidos no loader |
|---|---|---|---|
| AST | 1796 | 157 | 432 |
| CFG | 2181 | 234 | 22 |
| PDG | 2007 | 204 | 221 |

A perda de grafos no loader é outro problema, tratado em [testes_erro/falhas_loader/](testes_erro/falhas_loader/), que está em stand-by para já.

## 5. Limitações

- **Dataset completo:** os `.pt` do Linux da tese não estão nesta máquina. A amostra real foi reconstruída com a extração original, mas é uma amostra:
  - 160 ficheiros;
  - cerca de 10% de vulneráveis, contra 2,6% na tese;
  - por isso, as métricas absolutas não são comparáveis às tabelas.
- **Treino na amostra real:** 3 seeds em vez de 5; só CPU; sem afinação com Optuna (Tabela 5.8); sem o caminho AMP/VGG.
- **Experiência sintética:** funções simples e mais pequenas do que as reais; classes equilibradas; o GraphCodeBERT é substituído por uma soma de vetores fixos por token; não inclui a AST; uma única divisão treino/teste.
- **Diagnóstico do colapso:** usa só a seed 0.
- **Sonda linear:** só vê pares de nós ligados diretamente, enquanto a DGCNN agrega até 4 saltos.
- **Veredicto de `teste_aprofundado.py`:** o que o script imprime é texto fixo, não calculado a partir dos resultados.
- **Mecanismo:** é determinístico e depende só do formato dos ids, por isso aplica-se ao dataset completo, mesmo sem o ter.

## 6. Como correr

A partir de `verificacao_pipeline`:

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\teste_inicial.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\testes_erro\testes_mais_aprofundados\teste_aprofundado.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\testes_erro\kernel_real\kernel_real.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 01_alinhamento_arestas\testes_erro\testes_mais_aprofundados\diagnostico_sintetico.py
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe -X utf8 01_alinhamento_arestas\correcao\testar_correcao.py
```

- O `teste_inicial.py` e o `teste_aprofundado.py` precisam da preparação (00).
- Todos, exceto o `teste_inicial.py`, precisam do Joern.
- O `teste_aprofundado.py` pode demorar horas por causa da DGCNN; `--sem-dgcnn` salta essa parte.
- O `kernel_real.py` demora alguns minutos e o `diagnostico_sintetico.py` cerca de meia hora.
- O `testar_correcao.py` demora cerca de 5 minutos. Precisa da preparação (00) e da amostra real (passo 1 abaixo), mas não do Joern. Sai com código 1 se algum teste falhar.

Amostra real, por esta ordem, em `01_alinhamento_arestas\testes_erro\impacto_tese\`:

1. `construir_amostra.py --n-ficheiros 160 --workers 2` (cerca de 2 h);
2. `gerar_embeddings.py` (cerca de 50 min);
3. `verificar_estrutura.py`;
4. `correr_treinos.py 2` (cerca de 6 h para os 27 treinos);
5. `resumir.py`.
