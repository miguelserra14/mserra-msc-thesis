# Grafos que o loader descarta

> Cerca de 10% dos grafos nunca chegam ao modelo, nem com o código original nem com a correção do problema 01. Na AST perde-se 1 em cada 5, e **nenhuma função com `goto` entra no modelo**. Os grafos perdidos são sobretudo os das funções grandes, e por isso perde-se uma parte maior das vulneráveis: 34% na AST, contra 18% das restantes. Quase todos têm uma de duas causas, e em ambas pode-se recuperar o grafo sem perder informação. **Ainda não foi feita nenhuma alteração ao código**; este documento descreve o problema e o plano.

## O problema

O loader descarta um grafo inteiro quando uma aresta toca num nó que não tem vetor ([load_graphcodebert_embeddings.py:86](../../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L86); o original falhava no mesmo sítio, com outra mensagem). Um nó fica sem vetor quando não tem texto:

1. o `generate_txt_force` escreve `id código`, com o código tirado da etiqueta do `.dot` ([graph_extraction.py:47-50](../../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/graph_extraction.py#L47-L50));
2. se a etiqueta está vazia, a linha fica só com o id, e o `create_embeddings.py` deita-a fora ([create_embeddings.py:330](../../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L330));
3. o nó fica sem vetor, mas as arestas continuam a apontar para ele.

O código já previa texto vazio: o `generate_embeddings_batch` dá-lhe um vetor de zeros ([create_embeddings.py:112-117](../../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L112-L117)). Mas esses nós nunca lá chegam, porque são deitados fora no passo 2.

## Quanto se perde e porquê

Na amostra do kernel (160 ficheiros, 6659 grafos; ver [impacto_tese](../impacto_tese/)):

| Grafo | Grafos | Carregados | Descartados | Causa principal | Outros nós | Protótipos (problema 07) |
|---|---|---|---|---|---|---|
| AST | 2228 | 1796 (80,6%) | 432 | 388: rótulo do `goto` | 19 | 25 |
| CFG | 2203 | 2181 (99,0%) | 22 | — | — | 22 |
| PDG | 2228 | 2005 (90,0%) | 223 | 190: nó de fora da função | 8 | 25 |

**Causa 1, na AST: o rótulo do `goto`.** Em `goto ok;`, o Joern cria um nó filho com o rótulo (`ok`), e o `.dot` exporta-o com a etiqueta vazia (`label = <>`). Nos ids do Joern é o tipo 18. Na amostra:

- os 1577 nós destes são todos folhas: têm uma só aresta, que vem do nó do `goto`;
- o texto do rótulo já está no nó pai (`goto ok;`), por isso a aresta não acrescenta informação;
- **0 dos 1796 AST carregados têm um `goto`**. Como o kernel usa `goto` para tratar erros, a AST perde precisamente as funções com tratamento de erros.

**Causa 2, no PDG: arestas para nós de fora da função.** O PDG de uma função tem arestas DDG que vêm de nós que não estão declarados no seu `.dot`. São nós de outro sítio do ficheiro, por exemplo os inicializadores de uma tabela global (`{ NULL }`, `{ &cxusb_table[MEDION_MD95700], NULL }`), ligados por arestas como `DDG: <unknown> NULL`. Na amostra:

- os 584 nós destes são CALL ou METHOD (tipos 7 e 25) e só são origem de arestas;
- são dependências espúrias, que não pertencem ao grafo da função;
- num grafo típico são 1,2% das arestas (mediana), mas há grafos em que passam de metade (máximo 58%).

**Protótipos: os grafos sem nenhum nó com texto (~1%).** Em geral são as mesmas funções nos três tipos de grafo (por exemplo `eql_enslave`, `SEQ_printf`, `verbose`, `do_sigbus`). A definição tem uma macro do kernel (`__user`, `__printf(2, 3)`, `noinline`, `noinline_for_stack`, …), e sem os headers do kernel o Joern não a reconhece. Só fica o protótipo, um grafo de 5 nós todos com a etiqueta vazia (visto no `.dot` de `eql_enslave`), e é esse grafo que fica associado à função. **É o problema 07** ([RELATORIO](../../../menos_urgentes/07_funcoes_perdidas_pelo_joern/RELATORIO.md)) e não se resolve no loader. Na pipeline da tese, estes grafos nem chegam a dar um `.pt`, porque o `create_embeddings.py` salta os grafos sem texto ([create_embeddings.py:336](../../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L336)).

**Outros nós (19 AST, 8 PDG).** São blocos, identificadores e um tipo 30 sem texto. Ainda não se sabe o que são.

## Impacto

- **Os grafos perdidos não são aleatórios.** Perdem-se as funções grandes: na AST, a mediana dos descartados é de 199 nós, contra 46 nos carregados, e perde-se 74% das funções com mais de 400 nós. Para o mesmo tamanho, as duas classes perdem-se à mesma taxa. Mas como as funções vulneráveis são maiores, perde-se uma parte maior delas:

  | Grafo | Vulneráveis perdidas | Restantes perdidas |
  |---|---|---|
  | AST | 82 de 239 (34%) | 350 de 1989 (18%) |
  | PDG | 35 de 239 (15%) | 188 de 1989 (9%) |
  | CFG | 2 de 236 (1%) | 20 de 1967 (1%) |

- **A comparação entre tipos de grafo não usa as mesmas funções.** A AST é avaliada sem as funções grandes e sem nenhuma com `goto`; o CFG fica com quase todas.
- **Vale para a tese.** O código original descartava os mesmos grafos (mais 2 PDG carregados por acaso, com a estrutura errada; ver o T4). Em proporção, os números da amostra devem ser parecidos com os do dataset completo, mas isso não foi verificado. O [verificar_pt_tese.py](../../correcao/verificar_pt_tese.py) já conta os descartados na máquina da tese.

## Plano para resolver

**A regra proposta.** Em vez de descartar o grafo, tirar só as arestas que tocam num nó sem vetor, **e só nos dois casos conhecidos**:

| Caso | Como se reconhece num `.pt` | Porque se pode tirar a aresta |
|---|---|---|
| Rótulo do `goto` (AST) | nó sem vetor, folha, com uma só aresta de entrada, vinda de um nó cujo texto começa por `goto ` | o rótulo já está no texto do pai; não se perde nada |
| Nó de fora da função (PDG) | nó sem vetor, só origem de arestas, do tipo CALL ou METHOD | a aresta não pertence ao grafo da função |
| Qualquer outro | — | continua a descartar-se até se perceber o que é |

Assim a matriz de features não muda, e os 5982 grafos que já se carregam ficam bit a bit iguais. Todas as arestas que ficam ligam os nós certos.

**Alternativa considerada:** dar ao nó um vetor de zeros, que é a convenção do `generate_embeddings_batch`, e manter a aresta. Fica de lado para estes dois casos, porque acrescentaria à matriz um nó sem conteúdo (o rótulo) ou um nó que não é da função (no PDG). Pode servir para os 27 casos que sobram, quando se souber o que são.

**Passos:**

1. **Confirmar as causas no `.dot`, com um teste.** Tornar esta análise num teste em `falhas_loader/`: passar a amostra pelo Joern e confirmar que o tipo 18 é o rótulo do `goto` (`JUMP_LABEL`) e é sempre folha, e que os ids em falta no PDG nunca estão declarados no `.dot`. Identificar os 27 casos que sobram.
2. **Decidir com os orientadores.** A regra muda o dataset (na amostra, +388 AST e +190 PDG), e o modelo AST passa a ver funções com `goto` pela primeira vez.
3. **Implementar atrás de um segundo interruptor**, independente do do problema 01 (por exemplo `RECUPERAR_GRAFOS = False` por omissão), para se poder medir o efeito à parte. Com a mesma estrutura do 01:
   - o loader DGL aplica a regra no `aligned_edge_index`, em vez de dar erro;
   - o filtro do `EmbeddingDataset` usa a mesma regra;
   - a cache e as runs mudam de nome (por exemplo `-aligned-recuperados`), e o SVM/XGBoost recusa runs do modo errado;
   - os `.pt` não mudam, e não é preciso regenerar nada.
4. **Testar**, com as regras do 01: a verdade vem do `.dot`; há controlo negativo e mutantes.
   - Com o interruptor desligado, o resultado é bit a bit o atual.
   - Os 5982 grafos já carregados não mudam.
   - Nos grafos recuperados, as arestas que ficam são as do `.dot`. As que saem são só dos dois casos, verificado contra o `.dot` e não contra o código.
   - O `EmbeddingDataset` dá os mesmos grafos que o loader DGL.
5. **Medir.** Na amostra, quantos grafos se recuperam por tipo e por classe. Na máquina da tese, acrescentar ao `verificar_pt_tese.py` a contagem por causa.
6. **Treinar.** Entra no novo treino das Tabelas 5.4–5.8 como mais uma condição (corrigido + recuperados), para separar o efeito do do alinhamento.

**Resultado esperado na amostra**, sem contar os protótipos do problema 07:

| Grafo | Funções reais carregadas | |
|---|---|---|
| AST | 2184 de 2203 (99,1%) | era 81,5% |
| CFG | 2181 de 2181 (100%) | sem mudança |
| PDG | 2195 de 2203 (99,6%) | era 91,0% |

Os protótipos (~1%) só se recuperam com uma nova extração com os defines do kernel (problema 07).

## Limitações desta análise

- Foi feita uma vez, sobre os registos da amostra (`_gerado/impacto_tese/grafos`), e o Joern só foi passado de novo em 4 dos 160 ficheiros, para ver os `.dot`. Ainda não é um teste: é o passo 1 do plano.
- O tipo de nó vem dos bits altos do id do Joern (`id >> 32`). É estável numa versão do Joern, mas é um pormenor de implementação. Por isso a regra também pede a forma (folha com pai `goto`; nó só de origem).
- Na amostra há cerca de 10% de vulneráveis, contra 2,6% na tese. As taxas por classe valem para a amostra; o mecanismo (as funções grandes perdem-se mais) não depende disso.
