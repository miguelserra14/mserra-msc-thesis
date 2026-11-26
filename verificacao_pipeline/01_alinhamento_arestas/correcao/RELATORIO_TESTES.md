# Relatório: como foi testada a correção do problema 01 e o que se conclui

Este relatório explica, em detalhe, que testes foram feitos à correção do alinhamento entre nós e arestas (problema 01), o que cada um prova e o que não prova, e a que conclusões se chega. O erro em si está descrito em [../RELATORIO.md](../RELATORIO.md); a implementação da correção em [README.md](README.md).

Todos os números vêm das últimas execuções, guardadas em [resultado_correcao.txt](resultado_correcao.txt), [resultado_testes_adicionais.txt](resultado_testes_adicionais.txt) e [resultado_e2e.txt](resultado_e2e.txt).

---

## 1. Conclusões em resumo

1. **A correção põe cada aresta entre os nós certos.** Em 5982 grafos reais do kernel (AST, CFG e PDG), 507 909 de 507 909 arestas ligam os mesmos nós que no `.dot` do Joern, com a mesma direção e as mesmas repetições. O código original acerta 59 236 (11,7%).
2. **A correção não muda mais nada.** As features, a ordem dos nós e as etiquetas ficam bit a bit iguais. A única diferença, além das arestas, são 2 grafos PDG (em 2007) que o código original carregava por acaso, com a estrutura errada, e que passam a ser descartados.
3. **O código da Constança continua disponível e intacto.** Com o interruptor em `True`, o resultado é o do código original em todos os grafos testados, o que permite comparar com a tese.
4. **Não há outro caminho no repositório que use as arestas gravadas sem a correção.** O `EmbeddingDataset` (PyTorch Geometric) tinha o mesmo erro e está corrigido. O `fusion_all_graphs.py` e o caminho das features manuais estão alinhados por construção.
5. **O treino completo corre no modo corrigido** (ver secção 5).
6. **O que fica por provar:** que os `.pt` da tese (noutra máquina) foram gerados pelo mesmo código; e uma verificação independente feita por outra pessoa. Os 10% de grafos que o loader descarta e o texto dos nós da AST (problema 02) são problemas diferentes, que a correção não trata.

---

## 2. O que se queria provar

A correção tem uma tarefa simples: traduzir o `edge_index` gravado em cada `.pt` (numerado pela ordem de texto dos ids do Joern) para as linhas da matriz de features (ordenadas pela ordem numérica), usando o `node_to_idx` que cada `.pt` já guarda. Para se poder confiar nela, os testes tinham de responder a seis perguntas:

| # | Pergunta | Porque importa |
|---|---|---|
| P1 | Depois da correção, cada aresta liga os nós que liga no programa? | É a correção em si |
| P2 | A correção muda alguma coisa além das arestas? | Para que qualquer diferença nos resultados venha só do alinhamento |
| P3 | O modo original reproduz exatamente o código da Constança? | Para haver um baseline fiável e comparável com a tese |
| P4 | Os testes conseguem detetar uma correção errada? | Um teste que passa sempre não prova nada |
| P5 | Há caminhos no código que contornem a correção? | Uma cache antiga ou outro loader podiam anulá-la sem aviso |
| P6 | O treino real corre do princípio ao fim com a correção ligada? | Os testes anteriores verificam peças; este verifica a ligação entre elas |

---

## 3. Como os testes foram construídos

Quatro regras atravessam todos os testes. São elas que dão confiança aos resultados.

**3.1 A verdade não vem do código testado.** As arestas verdadeiras são lidas do `.dot` do Joern com um leitor próprio ([comum.py](../../comum.py), `ler_dot`), e não com a função da Constança nem com a correção. No teste A3, este leitor, a expressão regular da Constança e o pydot dão exatamente as mesmas arestas em 120 `.dot`.

**3.2 Sabe-se que nó está em cada linha da matriz.** Para comparar arestas como pares de nós, é preciso saber que nó do Joern ficou em cada linha da matriz de features. Usam-se duas técnicas independentes:

- *Features codificadas* (T1–T7, A1, A2): o vetor de cada nó é substituído por um número que identifica o nó. Depois de o loader construir o grafo, lê-se esse número em cada linha e sabe-se que id lá está, seja qual for a ordem que o loader usou.
- *Vetores reais do GraphCodeBERT* (A4, ponta a ponta): usam-se os vetores verdadeiros e identifica-se o texto de cada linha comparando o vetor. É a técnica mais próxima do uso real, porque não substitui nada.

**3.3 A comparação é exata.** Cada aresta é comparada como par (origem, destino) de ids do Joern, como multiconjunto: a direção conta e as arestas repetidas contam. Um grafo só conta como certo se o multiconjunto for igual ao do `.dot`.

**3.4 Há sempre um controlo negativo.** Cada teste corre também sobre o código original e tem de falhar. Se não falhasse, não teria poder para detetar o erro. Os testes de mutação (A1) levam esta ideia mais longe: estragam a correção de propósito e exigem que a bateria dê pelo estrago.

### Dados usados

| Conjunto | O que é | Onde entra |
|---|---|---|
| Ficheiro de teste | [dados/vuln.c](../../dados/vuln.c), 6 funções com casos difíceis, passado pelo Joern e pela extração original: 18 `.pt` reais | T0, T2, T7, A3, A5 |
| Amostra do kernel | 160 ficheiros do Linux nos commits exatos do dataset da tese, passados pela extração original: 6659 grafos (2228 AST, 2203 CFG, 2228 PDG), dos quais 5982 se carregam | T3–T7, A1, A4, ponta a ponta |
| Vetores reais | GraphCodeBERT real, aplicado ao texto de cada nó da amostra | A4, ponta a ponta |
| Grafos aleatórios | 20 000 grafos gerados (ids de 1 a 19 dígitos, laços, arestas repetidas, nós isolados, nós sem texto) | A2 |
| Código adversarial | Um ficheiro C com `"N" -> "M"` em strings e comentários, passado pelo Joern | A3 |

---

## 4. Os testes, um a um

### T0 — O código original está intacto e o interruptor funciona (P3)

- **Como:** compara-se, linha a linha, cada ficheiro do repositório com a versão do commit original (`c816995`). Corre-se a extração e o loader do repositório nos dois modos e compara-se o resultado com o do código original. Lê-se o interruptor em processos novos, com e sem variável de ambiente, e com ficheiros de interruptor mal escritos.
- **Resultado:**
  - Nos 5 ficheiros tocados, todas as linhas originais estão lá, pela mesma ordem; só há linhas acrescentadas (loader: 172 + 79; `detect_vulnerabilities.py`: 1180 + 6; `create_embeddings.py`: 353 + 73; `new_classifiers_svm_xgboost.py`: 271 + 8; `embedding_dataset.py`: 205 + 40).
  - No modo original, a extração dá os mesmos 18 `.pt` que o código original, e o loader dá os mesmos grafos em 6659 de 6659.
  - A variável de ambiente ganha ao ficheiro; um interruptor mal escrito dá erro em vez de escolher um modo em silêncio.
- **O que prova:** o baseline da tese continua disponível, e o interruptor não falha sem aviso.

### T1 — Casos feitos à mão (P1, P2)

- **Como:** 12 casos sintéticos, sem Joern: as duas causas do erro (ids de 11 e 12 dígitos; nós sem arestas), uma aresta repetida, um nó isolado, uma aresta para um nó sem texto, `node_to_idx` em falta ou com índices inválidos, grafo sem arestas, grafo sem nós.
- **Resultado:** 12/12. O código original falha o caso principal, como devia.
- **O que prova:** a correção trata os casos-limite. **Limite:** os casos foram pensados por quem escreveu a correção.

### T2 — Os 18 `.pt` reais do ficheiro de teste (P1)

- **Como:** carregam-se os 18 `.pt` escritos pela extração original e compara-se cada aresta com o `.dot`.
- **Resultado:** corrigido 328/328 arestas, 18/18 grafos exatos; original 50/328 (15%).
- **O que prova:** a correção funciona em ficheiros `.pt` verdadeiros, escritos pelo código da Constança, e não só em casos construídos.

### T3 — 5984 grafos reais do kernel (P1)

- **Como:** o mesmo que o T2, em escala, com features codificadas.
- **Resultado:**

  | Grafo | Carregados | Arestas certas | Grafos exatos |
  |---|---|---|---|
  | AST | 1796 | 134 788 / 134 788 | 1796 / 1796 |
  | CFG | 2181 | 119 358 / 119 358 | 2181 / 2181 |
  | PDG | 2005 | 253 763 / 253 763 | 2005 / 2005 |

- **O que prova:** a correção funciona em código real do kernel, nos três tipos de grafo, incluindo os casos raros que só aparecem em escala.

### T4 — Só as arestas mudam (P2)

- **Como:** nos grafos carregados pelos dois códigos, comparam-se os nós, o número de arestas e a matriz de features.
- **Resultado:** iguais em 5982/5982. Nenhum grafo novo é carregado. Dois PDG que o original carregava passam a ser descartados: `rsi_mac80211_attach` e `snd_toneport_source_info` têm uma aresta para um nó sem texto. O original carregava-os por acaso e com a estrutura errada; os outros 221 PDG com o mesmo defeito já eram descartados.
- **O que prova:** a matriz de features fica bit a bit igual. Isto importa porque o caminho AMP/VGG faz pooling por janelas de linhas e depende da ordem dos nós.

### T5 — `.pt` antigos e novos dão o mesmo grafo (P2)

- **Como:** gera-se o `.pt` com a extração corrigida a partir das mesmas arestas e compara-se o grafo com o do `.pt` antigo carregado pelo loader corrigido.
- **Resultado:** idênticos em 6659/6659 (e 18/18 a partir dos `.dot` reais).
- **O que prova:** não é preciso regenerar o dataset. Os `.pt` da tese, carregados pelo loader corrigido, dão o mesmo grafo que uma extração nova.

### T6 — Os grafos são os do treino já feito na amostra (P2)

- **Como:** compara-se o grafo do loader corrigido com o da condição "corrigido" usada para os números de treino em [../DETALHES.md](../DETALHES.md).
- **Resultado:** idênticos em 5982/5984 (os 2 restantes são os PDG do T4).
- **O que prova:** os números de treino da amostra já valem para esta correção (exatos na AST e no CFG; no PDG, aproximados por causa desses 2 grafos).

### T7 — O `EmbeddingDataset` do PyTorch Geometric (P1, P2, P3, P5)

- **Porquê:** o `embedding_dataset.py` tinha o mesmo erro. Não entra no treino da tese, mas é a entrada natural para trabalho futuro com o PyTorch Geometric, incluindo as suas ferramentas de XAI.
- **Como:** gravam-se os 6659 grafos da amostra como `.pt`, no formato da extração original, e compara-se o `EmbeddingDataset` com o loader DGL corrigido e com o `.dot`. Faz-se o mesmo com os 18 `.pt` reais.
- **Resultado:**
  - segue o mesmo interruptor (com e sem variável de ambiente, e quando é importado de fora da pasta `classification/`);
  - fica com os mesmos 5982 grafos que o loader DGL, com `x` e `edge_index` iguais, e 507 909/507 909 arestas certas;
  - no modo original, o resultado é o do código da Constança (6659 grafos, mesma ordem, mesma saída);
  - a subclasse `StratifiedEmbeddingDataset` herda o filtro.
- **O que revelou sobre o código original:** acertava só 45 971 de 658 269 arestas e devolvia **72 grafos vazios com etiqueta 0**. Quando um `.pt` falhava, o `except` do `get()` devolvia um grafo falso em vez de o descartar, ou seja, falsos negativos inventados. A correção tira esses grafos no `__init__`, com a mesma regra do loader DGL.

### A1 — Testes de mutação (P4)

- **Como:** estraga-se a correção de propósito e exige-se que a bateria de testes falhe. Primeiro, a bateria tem de passar com a correção verdadeira; sem isso, matar mutantes não provaria nada.
- **Mutantes da função de alinhamento** (7, todos apanhados): sem tradução (o erro original); linhas pela ordem de texto; origem e destino trocados; nó sem texto aceite em silêncio; nó sem texto mandado para a linha 0; `node_to_idx` com índices repetidos aceite; `edge_index` sem `node_to_idx` aceite.
- **Mutantes do `EmbeddingDataset`** (7, todos apanhados): `get()` sem alinhamento; `get()` com as linhas pela ordem de texto; `__init__` sem filtro; filtro sem verificar o alinhamento; filtro sem verificar os nós; modo original que alinha; modo original que filtra.
- **O que prova:** os testes têm poder para detetar os erros mais plausíveis numa correção deste tipo. **Limite:** os mutantes cobrem os erros em que pensei, não todos os possíveis.

### A2 — Fuzzing com 20 000 grafos aleatórios (P1)

- **Como:** grafos com 1 a 40 nós e até 120 arestas, ids de 1 a 19 dígitos, laços, arestas repetidas, nós isolados e, em 2675 grafos, nós sem texto. Testam-se `.pt` antigos e novos.
- **Resultado:** 20 000/20 000 exatamente certos (arestas, ordem, direção, repetições) ou descartados quando devem. O original acerta 7327 dos 17 325 grafos válidos que carrega, e carrega 1081 grafos com nós sem texto.
- **O que prova:** a correção não depende de particularidades do Joern atual, como os ids terem 11 ou 12 dígitos. Cobre formatos que outra versão do Joern pudesse gerar.

### A3 — Arestas falsas no `.dot` (P1)

- **Porquê:** a leitura das arestas usa uma expressão regular sobre o ficheiro inteiro. Se o código C de um nó contivesse um padrão como `"123" -> "456"`, podia ser lido como uma aresta.
- **Como:** passa-se pelo Joern um ficheiro C com esse padrão em strings e comentários e comparam-se três leitores do `.dot`.
- **Resultado:** o texto chega ao `.dot`, mas com as aspas escapadas como `&quot;`, por isso nunca forma uma aresta. Os três leitores dão o mesmo resultado em 120/120 `.dot`.
- **O que prova:** o risco não existe, pelo menos nesta versão do Joern.

### A4 — Vetores reais do GraphCodeBERT (P1)

- **Como:** nos 6659 grafos da amostra, com os vetores verdadeiros, identifica-se o texto de cada linha da matriz pelo vetor e compara-se cada aresta, como par de textos, com o `.dot`.
- **Resultado:** cada linha é identificada pelo seu vetor (0 linhas ambíguas); 507 909/507 909 arestas ligam os textos certos; o original liga 59 236.
- **O que prova:** confirma o T3 sem substituir os vetores por números, ou seja, nas condições exatas do treino. Como compara textos, não distinguiria dois nós com o mesmo texto; o T3, que compara ids, cobre esse caso.

### A5 — Caminhos que contornam a correção (P5)

- **Como:** procuram-se todos os ficheiros do repositório que leem `.pt`. Os que usam o `edge_index` gravado têm de aplicar a correção e estar testados. Os que reconstroem as arestas a partir do `.dot` são verificados à parte. Corre-se também o SVM/XGBoost nos dois modos.
- **Resultado:**
  - só dois ficheiros leem `.pt` além do loader e da extração: o `embedding_dataset.py` (corrigido, testado no T7) e o `fusion_all_graphs.py`;
  - o `fusion_all_graphs.py` não usa o `edge_index` gravado: constrói as features e as arestas a partir do `.dot` com a mesma ordem de nós. Num teste funcional, as arestas batem com o `.dot` em 18/18 grafos reais (comparadas como pares de vetores), e a mesma comparação falha em 17 com o loader original;
  - nenhum destes dois ficheiros é importado pelo treino;
  - no modo corrigido, o SVM/XGBoost recusa a run da tese (desalinhada) em vez de a ler em silêncio.
- **O que prova:** não há, no repositório, um caminho conhecido que use as arestas gravadas sem a correção.

Fora dos testes, li também o caminho das features manuais (Tabela 5.3). A matriz de adjacência e a de features usam a mesma lista de nós ([graph_feature_extraction.py:182-189](../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/graph_feature_extraction.py#L182-L189)), por isso este caminho nunca teve o erro. Isto foi verificado por leitura, não por um teste.

---

## 5. O teste de ponta a ponta (P6)

<!-- E2E: preenchido no fim do teste -->
*Em curso.*

---

## 6. Conclusões e grau de confiança

| Afirmação | Confiança | Base |
|---|---|---|
| O loader corrigido põe cada aresta entre os nós certos, nos `.pt` gerados pelo código da Constança | **Muito alta** | T2, T3, A4 (507 909/507 909, com features codificadas e com vetores reais), A2 (20 000 grafos aleatórios), controlos negativos e mutação |
| A correção não muda nada além das arestas (e de 2 PDG em 2007) | **Muito alta** | T4, T5, T6: comparações bit a bit |
| O modo original é o código da Constança | **Muito alta** | T0: linhas intactas, resultados iguais em 6659/6659 grafos |
| Não há outro caminho no repositório que contorne a correção | **Alta** | A5 e T7; depende de a pesquisa ter encontrado todos os leitores de `.pt` |
| O treino completo corre no modo corrigido, sem usar a cache antiga | Ver secção 5 | Teste de ponta a ponta |
| A correção serve para os `.pt` da tese | **Alta, não verificada** | Os `.pt` da tese não estão nesta máquina. Se algum veio de outra versão do código, é realinhado mal sem erro. O [verificar_pt_tese.py](verificar_pt_tese.py) deteta-o. |

---

## 7. O que os testes não cobrem

1. **Os `.pt` da tese.** É o risco que não se consegue eliminar nesta máquina. Basta correr o [verificar_pt_tese.py](verificar_pt_tese.py) na máquina que os tem: confirma que a numeração de cada `.pt` é uma das duas que o código conhece e, se os `.dot` estiverem lá, compara as arestas com eles.
2. **Uma verificação independente.** Os testes e a correção foram escritos pelo mesmo autor. Se este tivesse percebido mal o que é uma aresta certa, os dois concordariam no erro. A mutação (que inclui trocar origem e destino) e o A4 (que compara textos) reduzem este risco. Uma verificação à mão por outra pessoa elimina-o: escolher uma função, imprimir as arestas do grafo carregado como pares de texto e compará-las com o `.dot`.
3. **A representatividade da amostra.** São 160 ficheiros do kernel, passados por uma versão do Joern. O mecanismo do erro e da correção é determinístico, e o fuzzing cobre formatos de ids que não aparecem na amostra, por isso o risco é baixo.
4. **Os grafos descartados.** Cerca de 10% dos grafos (677 de 6659: 432 AST, 22 CFG, 223 PDG) não se carregam, nem no código original nem no corrigido, porque uma aresta toca num nó sem texto. Na AST é quase sempre o rótulo de um `goto` (nenhuma função com `goto` se carrega); no PDG, um nó de fora da função; os restantes ~1% são protótipos (problema 07). As arestas dos grafos carregados estão certas, mas estes grafos não estão no modelo. É outro problema, com causas e plano em [../testes_erro/falhas_loader/RELATORIO.md](../testes_erro/falhas_loader/RELATORIO.md).
5. **O texto dos nós.** Na AST, os nós IDENTIFIER e LITERAL recebem o texto da expressão onde estão (`a` passa a `a + b`). As arestas estão certas, mas o conteúdo de parte dos nós não é o do nó. É o problema 02, que a correção não trata.
6. **As métricas.** Nenhum teste mede o desempenho no dataset completo. Os números de treino da amostra estão em [../DETALHES.md](../DETALHES.md); o teste de ponta a ponta usa 300 grafos e 1 run por configuração e serve para verificar a ligação entre as peças, não para medir.

---

## 8. Como reproduzir

A partir de `verificacao_pipeline`, com o ambiente `graphcodebert`:

```powershell
python -X utf8 01_alinhamento_arestas/correcao/testar_correcao.py      # T0–T7, ~10 min
python -X utf8 01_alinhamento_arestas/correcao/testes_adicionais.py    # A1–A5, ~15 min
python -X utf8 01_alinhamento_arestas/correcao/e2e_treino.py --n 300 --runs 1   # ponta a ponta, ~3 h
python verificar_pt_tese.py <pasta dataset_final_embeddings> --dots <pasta extraction/>   # na máquina da tese
```

Os testes precisam da preparação (`00_preparacao/preparar.py`) e da amostra do kernel (`testes_erro/impacto_tese/construir_amostra.py` e `gerar_embeddings.py`).
