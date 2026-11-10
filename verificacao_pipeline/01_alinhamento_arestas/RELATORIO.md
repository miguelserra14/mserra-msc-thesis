# 01 — Os textos dos nós ficam no sítio errado do grafo

> A pipeline ordena os nós de uma maneira para as arestas e de outra para as features. Em praticamente todos os grafos, cada nó fica com o texto de outro nó: a forma do grafo está certa, mas só 9 a 15% das arestas ligam os fragmentos de código certos. O modelo treinado não usa a representação que a tese descreve. A correção já está aplicada e acerta 100% das arestas. Falta voltar a treinar no dataset completo.

## O erro

Os nós de cada grafo são guardados em **duas ordens diferentes**:

- as **arestas** numeram os nós pela ordem dos ids do Joern **como texto** ([create_embeddings.py:168](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L168));
- as **features** (o vetor GraphCodeBERT de cada nó) são empilhadas pela ordem **numérica** ([load_graphcodebert_embeddings.py:118](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L118)).

As duas ordens divergem porque os ids do Joern têm 11 ou 12 dígitos, e como texto `"107374182400"` vem antes de `"30064771072"`. Todas as funções têm ids dos dois tamanhos, por isso a troca acontece em quase todos os grafos.

O resultado é que a aresta "nó 3 → nó 7" liga, nas features, dois nós diferentes dos que liga no programa. Exemplo, no CFG de uma função `compute`:

| Nó real | Texto que a GNN vê nesse nó | Texto que lhe chega pelas arestas | Devia chegar |
|---|---|---|---|
| `r > 10` | `compute` | `r = 10` | `r = a + b` |
| `r = 10` | `int` | `compute` | `r > 10` |
| `helper(r)` | `return helper(r);` | `compute`, `int` | `r > 10`, `r = 10` |

## A prova em código real

Em 5984 grafos de 160 ficheiros do kernel, sorteados do dataset da tese e passados pela extração original, cada aresta foi comparada com o `.dot` do Joern:

| Grafo | Estrutura igual à do `.dot` | Nós com o texto de outro nó | Arestas que ligam os textos certos |
|---|---|---|---|
| AST | 1796 de 1796 | 98% | 14,8% |
| CFG | 2181 de 2181 | 97% | 14,1% |
| PDG | 2005 de 2007 | 96% | 8,8% |

## Impacto

- **Na representação.** A tese descreve uma GNN que aprende relações entre instruções: quem controla quem (CFG), quem depende de quem (PDG) e quem contém quem (AST). Com o erro, 85 a 91% dessas relações são falsas. Isto vale para todos os resultados GraphCodeBERT + DGCNN (Tabelas 5.4 a 5.8). A Tabela 5.3 usa features manuais e não é afetada. O conjunto de textos de cada grafo não muda, por isso o que depende só de que instruções existem (por exemplo, se há um `memcpy`) continua a funcionar.
- **Na aprendizagem.** Numa experiência controlada com funções sintéticas, em que as classes só diferem na ordem das instruções, a DGCNN com o PDG corrigido chega a 74% (mediana de 5 seeds). Com o erro, fica nos 42%, o mesmo que prever sempre a mesma classe. Quando as classes diferem no conteúdo, o erro não tem efeito.
- **Nos resultados.** Repetiu-se o caminho da Tabela 5.7 numa amostra real, com o GraphCodeBERT real, em três condições que só diferem nas arestas. F1 no teste (média de 3 seeds; intervalo entre AST, CFG e PDG):

  | Classificador | Com o erro | Corrigido | Sem arestas |
  |---|---|---|---|
  | SVM | 0,22–0,27 | 0,23–0,28 | 0,24–0,29 |
  | XGBoost | 0,13–0,16 | 0,18–0,24 | 0,23 |

  - No SVM, o erro não tem efeito visível. No XGBoost, custa cerca de 0,05 a 0,10 de F1.
  - **O modelo sem arestas fica ao mesmo nível do corrigido, ou acima.** Nesta configuração a DGCNN não tira proveito da estrutura, com ou sem o erro.

## Gravidade

| Pergunta | Gravidade | Porquê |
|---|---|---|
| O modelo usa a representação descrita? | **Crítica** | Não: 85 a 91% das arestas ligam os textos errados, em quase todos os grafos. |
| Os números das Tabelas 5.4–5.8 estão errados? | **Moderada** | Provavelmente um pouco abaixo do que dariam com a correção, mas na mesma ordem de grandeza. |
| As conclusões sobre a estrutura (AST vs CFG vs PDG, papel da GNN) são válidas? | **Alta** | Com o erro, nenhuma comparação entre tipos de grafo mede a estrutura. E mesmo corrigido, o modelo sem arestas fica ao mesmo nível, por isso a tese não mostra que a estrutura ajuda. Isto não se resolve só com esta correção. |

## Correção (aplicada)

- **Loader:** renumera as arestas para a ordem das features, com o mapeamento que cada `.pt` já guarda. **Não é preciso regenerar nenhum `.pt`.** As features e a sua ordem não mudam; só mudam as arestas.
- **Extração:** os `.pt` novos já saem certos, incluindo os nós isolados.
- **Treino:** a cache dos grafos muda de nome, para que uma cache antiga não anule a correção sem aviso.

O código da Constança não foi apagado nem alterado; só foram acrescentadas linhas. Comparado com o GitHub dela, o código de partida só difere em caminhos absolutos, e as linhas do erro são iguais (ver [correcao/README.md](correcao/README.md)).

Para alternar entre o código dela e o corrigido, muda-se uma variável em [correcao/interruptor.py](correcao/interruptor.py):

```python
USAR_CODIGO_ORIGINAL = True    # código original da Constança (baseline, como na tese)
USAR_CODIGO_ORIGINAL = False   # corrigido
```

Numa máquina sem esta pasta, usa-se a variável de ambiente `LEGACY_EDGE_ALIGNMENT=1` ou `=0`.

**Verificação:** na amostra real, 507 909 de 507 909 arestas ficam certas, e o modo original reproduz o código dela bit a bit. Os detalhes da implementação, os 7 testes e o que pode passar por baixo do radar estão em [correcao/README.md](correcao/README.md).

## O que falta fazer

1. Usar o loader novo na máquina com os `.pt` do Linux da tese.
2. Confirmar numa amostra que as arestas batem com os `.dot`.
3. Voltar a treinar as Tabelas 5.4–5.8 com a correção **e sem arestas**. Sem esta comparação não se pode dizer se a estrutura ajuda.

## Limitações

- Os números de treino vêm de uma amostra (160 ficheiros, cerca de 10% de vulneráveis contra 2,6% na tese), não do dataset completo. O mecanismo do erro é determinístico, por isso a sua existência aplica-se ao dataset completo; a dimensão do efeito nas métricas não.
- O treino na amostra usou 3 seeds, sem Optuna e sem o caminho AMP/VGG.
- O treino da DGCNN colapsa muitas vezes numa só classe, com ou sem o erro. É outro problema, ainda sem causa provada.

## Mais detalhe

- [DETALHES.md](DETALHES.md): todas as tabelas (sonda linear, DGCNN sintética, F1 e PR-AUC por grafo), como foi testado, a amostra real, todas as limitações e como correr.
- [correcao/README.md](correcao/README.md): o que mudou em cada ficheiro e os testes da correção.
- [testes_erro/falhas_loader/](testes_erro/falhas_loader/): grafos que o loader descarta (outro problema, em stand-by).
