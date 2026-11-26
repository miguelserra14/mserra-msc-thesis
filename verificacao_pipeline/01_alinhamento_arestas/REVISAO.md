# Revisão: a pipeline, o problema dos nós, a correção e os testes

Documento de apoio para explicar o trabalho feito no problema 01. Os números de linha referem-se ao código atual do repositório da Constança.

---

## 1. A pipeline

A pipeline transforma funções C do kernel Linux em grafos com texto em cada nó e treina uma GNN (DGCNN) para dizer se a função é vulnerável. Tem duas fases: **extração** (pasta `extraction/`) e **classificação** (pasta `classification/`).

```
linux-functions.csv ──► [1] checkout do commit ──► [2] Joern ──► N-{tipo}.dot ──► [3] N-{tipo}.dot.txt
 (função, commit,                                              (1 por função)     (id + código de cada nó)
  vulnerável?)                                                       │
                                                                     ▼
                         [4] functions-{tipo}-linux.csv ──► [5] ...-filtered.csv
                             (função → .dot, etiqueta)          (o que entra no modelo)
                                                                     │
                                                                     ▼
                         [6] create_embeddings.py ──► {função}.pt  (vetores dos nós + arestas)
                                                                     │
══════════════════════════════════ classificação ════════════════════╪══════════════════════
                                                                     ▼
                         [7] loader ──► grafos DGL ──► cache .pkl ──► [8] DGCNN ──► embeddings do grafo
                                                                                      │
                                                                     [9] SMOTETomek + SVM / XGBoost
                                                                                      │
                                                                             Tabelas 5.4–5.8
```

| Passo | Código | Entrada | Saída (onde aparece) |
|---|---|---|---|
| 0. Dataset de origem | — (dataset de João Henggeler) | — | `extraction/function-data/linux-functions.csv`: uma linha por (ficheiro, commit), com a lista de funções em JSON (`Name`, `Lines`, `Vulnerable`) |
| 1. Checkout | `main.py` → `version_management.load_commit` | o commit vulnerável | o repositório do Linux no estado desse commit |
| 2. Grafos | `graph_extraction.extract_graph_per_file` (`joern-parse` + `joern-export --repr ast/cfg/pdg`) | cada ficheiro `.c` | `output/linux/output-{tipo}-{commit}/{caminho com ---}/N-{tipo}.dot`, um por função (mais `<global>` e operadores) |
| 3. Texto dos nós | `graph_extraction.generate_txt_force` | cada `.dot` | `N-{tipo}.dot.txt` ao lado: uma linha `id código` por nó (a última parte da etiqueta do Joern) |
| 4. Mapeamento | `convert_dot_to_json.map_single_commit_to_graph` | CSV de origem + `.dot` | `output-data/functions-{tipo}-linux.csv`: `commit;ficheiro;função;.dot;vulnerável`. O mapeamento é feito só pelo nome da função. |
| 5. Filtragem | passo manual (não há script) | o CSV anterior | `functions-{tipo}-linux-filtered.csv`, com a coluna `CFG_filepath` |
| 6. Embeddings | `create_embeddings.py` | cada linha do CSV filtrado: o `.txt` e o `.dot` | `dataset_final_embeddings/{tipo}/….pt` e `dataset_manifest.csv` |
| 7. Loader | `load_graphcodebert_embeddings.py` | os `.pt` | grafos DGL num DataFrame (`graphs`, `label`), em cache em `datasets_embeddings/{tipo}-graphcodebert-embeddings[-aligned].pkl` |
| 8. DGCNN | `detect_vulnerabilities.py` | a cache | `output_baseline/runs/{configuração}/run_N/`, com `dgcnn_embeddings_{train,test}_after.pt` e as etiquetas |
| 9. Classificadores | `new_classifiers_svm_xgboost.py` | as embeddings do passo 8 | as métricas das Tabelas 5.4–5.8 |

**Como são os ficheiros** (exemplos reais da pipeline de teste):

- **`.dot` do Joern:** uma linha por nó, `"107374182400" [label = <METHOD, 1<BR/>…>]`, e uma por aresta, `"176093659136" -> "120259084288"`. Os ids têm 11 dígitos (CALL, IDENTIFIER, BLOCK, …) ou 12 (METHOD, METHOD_RETURN, PARAM, RETURN, …).
- **`.dot.txt`:** `176093659136 struct item { int len; char *data; }`, ou seja, o id e o código do nó.
- **`.pt`** (um por função): é aqui que está o problema 01.
  ```
  embeddings     dict id → vetor de 768 (GraphCodeBERT do código de cada nó, mean pooling)
  edge_index     tensor [2, nº arestas], com as arestas como pares de índices
  node_to_idx    dict id → índice usado no edge_index
  label, function_name, dot_path, commit, filepath, graph_type, num_nodes, num_edges, pooling_strategy
  ```

**O treino (passos 8 e 9), em resumo:**

- Por cada seed, divide-se o conjunto em 70% de treino e 30% de teste, de forma estratificada.
- A DGCNN tem 4 camadas GraphConv de 32 dimensões e SortPooling com k=32.
- O classificador é Conv1D (T9) ou VGG com Adaptive Max Pooling (T10); o T11 e o T12 juntam um autoencoder. Treina 30 épocas, com pesos por classe.
- As embeddings do encoder treinado vão depois para o SVM e o XGBoost, com SMOTETomek no treino.

**Onde está o problema 01:** entre os passos 6 e 7. O passo 6 numera as arestas de uma maneira e o passo 7 ordena as features de outra.

---

## 2. O problema dos nós: impacto e como foi medido

**Numa frase:**

- o passo 6 numera os nós das arestas por ordem de **texto** dos ids ([create_embeddings.py:168](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L168)) e só conta os nós que têm arestas;
- o passo 7 empilha as features por ordem **numérica** ([load_graphcodebert_embeddings.py:118](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L118)).

Como os ids têm 11 e 12 dígitos, as duas ordens divergem em praticamente todos os grafos.

**O impacto, em três níveis:**

| Nível | O que acontece | Evidência |
|---|---|---|
| **Representação** | A forma do grafo fica certa, mas os textos estão nos nós errados: é uma rotação. Cada nó recebe o texto do nó `m` posições à frente, em que `m` é o nº de ids de 12 dígitos. | Em 5984 grafos reais: 96–98% dos nós têm o texto de outro nó e só 9–15% das arestas ligam o par de instruções certo. |
| **Aprendizagem** | Com o erro, a GNN deixa de conseguir aprender relações de ordem entre instruções. O conteúdo ("há um `memcpy`?") não é afetado. | 600 funções sintéticas. No PDG, a sonda linear cai de 99,7% para 67,7% e a DGCNN de 74% para 42% (o nível de prever sempre a mesma classe). No conteúdo, 100% nos dois casos. |
| **Resultados** | As métricas da tese provavelmente estão um pouco abaixo do que dariam com a correção. | Na amostra real, o XGBoost ganha 0,05–0,10 de F1 com a correção; o SVM não muda. |

**A conclusão mais importante não vem do erro em si.** Mesmo corrigido, o modelo **sem arestas** fica ao mesmo nível do modelo com arestas, ou acima. Com o erro, nenhuma comparação AST vs CFG vs PDG media a estrutura. Depois de corrigido, a estrutura continua a não acrescentar informação nesta configuração. A tese não demonstra que a estrutura ajuda.

**Os testes que mostram o erro** (pasta `testes_erro/` e `teste_inicial.py`), do mais simples ao mais próximo da tese:

| Teste | Pergunta | Porque foi preciso |
|---|---|---|
| `teste_inicial.py` (18 grafos de um ficheiro C de teste) | O erro existe? | É a prova mínima: corre o código original e compara aresta a aresta com o `.dot`. Dá 15% de arestas certas. |
| `kernel_real/` (135 funções do kernel) | Existe em código real? | O ficheiro de teste podia ser atípico. No kernel, 7–12% de arestas certas, e a rotação é exata. |
| `impacto_tese/verificar_estrutura.py` (5984 grafos) | A forma do grafo muda, ou só os textos? | Distingue "grafo errado" de "textos trocados". A forma fica certa em 5982/5984. |
| `testes_mais_aprofundados/` (600 funções sintéticas) | O erro impede a aprendizagem? | Em dados reais não se sabe o que deveria ser aprendido. Aqui, as tarefas foram construídas de propósito para separar conteúdo de estrutura. |
| `impacto_tese/treinar.py` (5984 grafos reais, GraphCodeBERT real, 3 seeds) | Quanto muda nos resultados? | É o caminho da Tabela 5.7, em três condições que só diferem nas arestas (com o erro, corrigido, sem arestas). A condição "sem arestas" é o controlo: mede quanto a estrutura vale. |

---

## 3. A correção

### Como funciona

O `.pt` já guarda o `node_to_idx`, ou seja, a numeração exata que o `edge_index` usou. O loader corrigido usa-o para traduzir cada aresta para as linhas da matriz de features:

```
features (ordem numérica):   linha 0 = 30064771072   linha 1 = 68719476736   linha 2 = 107374182400
node_to_idx (ordem de texto): 107374182400 → 0       30064771072 → 1         68719476736 → 2
aresta guardada (0 → 1)  = 107374182400 → 30064771072
aresta corrigida         = linha 2 → linha 0
```

Em código ([load_graphcodebert_embeddings.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py), função `aligned_edge_index`): constrói-se uma tabela índice → linha e aplica-se a todo o `edge_index` de uma só vez.

### Onde aparece

| Ficheiro | O que se acrescentou |
|---|---|
| `classification/load_graphcodebert_embeddings.py` | `aligned_edge_index` e a chamada a esta função logo depois de ler o `edge_index`. **É esta a correção que importa.** |
| `extraction/create_embeddings.py` | `parse_dot_edges_aligned`: os `.pt` novos já saem com ordem numérica e com todos os nós |
| `classification/detect_vulnerabilities.py` | A cache de grafos passa a chamar-se `…-aligned.pkl` no modo corrigido |
| `classification/embedding_dataset.py` | O mesmo erro na versão PyTorch Geometric do loader. Usa a `aligned_edge_index` e o interruptor do loader, e no `__init__` tira os grafos que não se podem alinhar (o `get()` devolvia-os como grafos vazios com etiqueta 0). Não entra no treino da tese. |
| `classification/new_classifiers_svm_xgboost.py` | Recusa ler embeddings de uma run que não corresponde ao modo do interruptor |
| `verificacao_pipeline/01_alinhamento_arestas/correcao/interruptor.py` | `USAR_CODIGO_ORIGINAL = True/False`, lido pelos três ficheiros |

### Escolhas de design e porquê

| Escolha | Alternativas consideradas | Porque se escolheu esta |
|---|---|---|
| **Corrigir no loader, a partir do `node_to_idx`** | Regenerar o `edge_index` de todos os `.pt` a partir dos `.dot` | Os `.pt` da tese não estão nesta máquina e regenerá-los obriga a ter todos os `.dot`. O loader corrige qualquer `.pt` existente sem tocar no dataset. |
| **Mudar as arestas, não as features** | Reordenar as features para a ordem das arestas (foi a primeira versão) | O caminho AMP/VGG faz pooling por janelas de linhas, por isso depende da ordem dos nós. Reordenar as features mudava esse caminho e fazia `.pt` antigos e novos darem grafos diferentes. Assim, a matriz de features fica bit a bit igual à original. |
| **Corrigir também a extração** | Deixar a extração como está | Os `.pt` novos ficam certos mesmo com o loader original, e passam a incluir os nós isolados. Os dois caminhos dão o mesmo grafo. |
| **Só linhas acrescentadas** | Editar as funções dela | O código original continua todo lá, comentários incluídos. Isto é verificável, e a comparação com a tese continua possível. |
| **Um interruptor num ficheiro, com variável de ambiente como alternativa** | Só variável de ambiente; voltar a um commit antigo | Mudar uma linha é simples e fica registado. A variável serve na máquina do dataset completo, onde esta pasta não existe. |
| **Casos-limite iguais ao original** | Recuperar os grafos que o loader perdia | Assim, a diferença nos resultados vem só do alinhamento. Recuperar grafos é outro problema (`falhas_loader`). A única mudança são 2 PDG que o original carregava por acaso, mal; agora são descartados como os outros 221 com o mesmo defeito. |
| **Falhar com erro quando não se pode alinhar** (`node_to_idx` em falta, numeração inválida, interruptor mal escrito) | Escolher um comportamento por omissão | Um grafo mal alinhado sem aviso foi exatamente o erro original. |
| **No `EmbeddingDataset`, filtrar no `__init__`** | Só alinhar no `get()`; dar erro no `get()` | O `get()` não pode descartar um grafo: o seu `except` devolvia um grafo vazio com etiqueta 0, e cada grafo que não se pode alinhar virava um falso negativo em silêncio. Dar erro parava o treino a meio de uma época. Filtrar no `__init__` deixa o dataset com os mesmos grafos que o loader DGL. |
| **Cache com nome diferente** | Manter o nome | O `detect_vulnerabilities.py` reutiliza a cache se ela existir. Com o mesmo nome, uma cache antiga anulava a correção sem aviso. |
| **Scripts de verificação presos ao código do git** | Usar o código do repositório | Os testes do erro continuam a medir a pipeline da tese, mesmo depois de corrigida. |

---

## 4. Testes da correção e resultados

Script: `correcao/testar_correcao.py`. Tem 52 verificações e passam todas.

**O que torna os testes fiáveis:**

- **Verdade independente:** as arestas verdadeiras vêm do `.dot`, lido com um parser próprio, e não do código testado.
- **Comparação aresta a aresta:** sabe-se que nó está em cada linha da matriz, por isso cada aresta é comparada como um par de ids do Joern.
- **Comparações exatas:** as arestas repetidas contam.
- **Controlo negativo:** o mesmo teste corre no código original e tem de falhar.

| Teste | O que verifica | Porque é necessário | Resultado |
|---|---|---|---|
| **T0** | O código dela está intacto; o modo original reproduz o código dela; o interruptor funciona | Sem isto não há baseline fiável para comparar com a tese | 0 linhas originais alteradas (1180 + 172 + 353 mantidas). Modo original igual ao original em 6659/6659 grafos e 18/18 `.pt` extraídos. O interruptor é lido corretamente, e um ficheiro mal escrito dá erro. |
| **T1** | Casos sintéticos feitos à mão (as duas causas, uma aresta repetida, 5 casos-limite) | A correção podia funcionar só nos casos normais | 12/12. O original falha, como devia. |
| **T2** | Os 18 `.pt` reais escritos pela extração dela | Os sintéticos podiam não representar o Joern real | 328/328 arestas certas (original: 50/328); mesmas labels e features |
| **T3** | 5984 grafos reais do kernel | Os casos raros só aparecem em escala | **507 909/507 909 arestas certas** |
| **T4** | Só as arestas mudam | Para isolar o efeito e não afetar o VGG | Features, nós e ordem iguais. Encontrou os 2 PDG descartados. |
| **T5** | Um `.pt` antigo e um `.pt` novo dão o mesmo grafo | É o que permite não regenerar o dataset | Idênticos em 6659/6659 |
| **T6** | O loader corrigido dá os grafos já usados no treino da amostra | Para os números da secção 2 valerem para esta correção | Iguais bit a bit em 5982/5984. Na AST e no CFG os números são exatos; no PDG são uma aproximação. |
| **T7** | O `EmbeddingDataset` (PyTorch Geometric) segue o interruptor e dá os mesmos grafos que o loader DGL | É a entrada natural para XAI com o PyTorch Geometric; tinha o mesmo erro | Mesmos 5982 grafos, `x` e `edge_index` iguais, 507 909/507 909 arestas certas. O original devolvia 72 grafos vazios com etiqueta 0. O modo original é o código dela. |

Em [testes_adicionais.py](correcao/testes_adicionais.py) há mais cinco testes: mutação (A1, 7 mutantes da correção e 7 do `EmbeddingDataset`), fuzzing com 20 000 grafos (A2), Joern com código adversarial (A3), vetores reais do GraphCodeBERT (A4) e caminhos que contornam a correção (A5, que confirma que o `fusion_all_graphs.py` reconstrói as arestas a partir do `.dot` e está alinhado).

**O que ainda não está provado:**

1. **Os `.pt` da tese.** Se algum foi gerado por outra versão do código, com um `node_to_idx` que não corresponde às arestas, seria realinhado mal sem dar erro. Só se resolve repetindo o T3 nos dados reais.
2. **O treino com o loader corrigido.** Os grafos são iguais aos já treinados, mas não houve um treino novo a confirmá-lo.
3. ~~**Arestas falsas no `.dot`.**~~ Fechado pelo A3: o Joern escapa as aspas do código como `&quot;`, por isso um padrão como `"123" -> "456"` no código C nunca chega ao `.dot` como aresta (pelo menos nesta versão do Joern).

**Próximos passos:**

1. Repetir o T3 nos `.pt` da tese.
2. Voltar a treinar as Tabelas 5.4–5.8 em três condições: original, corrigido e sem arestas.
3. Decidir o que a tese pode afirmar sobre a estrutura.
4. Estabilizar o treino da DGCNN, que colapsa muitas vezes numa só classe.
5. Recuperar os grafos que o loader perde.
