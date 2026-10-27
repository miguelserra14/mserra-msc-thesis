# 13 — Informação que se perde entre o grafo e a GNN

**Veredicto: PROBLEMA CONFIRMADO — limitações de desenho, não erros · Gravidade: BAIXA**

> Há três limitações:
> - perde-se o tipo de aresta do PDG (controlo vs. dados);
> - na AST, a informação só flui da raiz para as folhas;
> - o GraphCodeBERT é usado como codificador de texto nó a nó, sem o data-flow que o relatório lhe atribui.

## Pergunta

Que partes da representação do Joern e do GraphCodeBERT chegam de facto à DGCNN?

## Como o teste funciona

- **(a)** Conta os tipos de aresta nos `.dot` do PDG do ficheiro de teste, e verifica o que o `.pt` guarda.
- **(b)** Mostra com um grafo mínimo de 2 nós em que sentido o `GraphConv` do DGL propaga a informação, e conta as arestas que entram na raiz da AST.
- **(c)** Inspeciona como `create_embeddings.py` chama o tokenizer e o modelo.

## Resultado

- **(a)**
  - Arestas do PDG: 119 DDG (dados), 14 CDG (controlo), 1 sem etiqueta.
  - O `.pt` guarda `edge_index`, mas nenhum campo com o tipo de aresta.
  - 15 de 134 arestas repetem um par de nós (uma por variável), e o `GraphConv` conta-as mais do que uma vez.
- **(b)**
  - No grafo `0 → 1`, mudar as features do nó 1 **não** altera a saída do nó 0.
  - Na AST de `compute`, entram 0 arestas na raiz (METHOD).
- **(c)**
  - `tokenizer(valid_codes, …)` recebe a lista de textos de cada nó separadamente.
  - `model(**inputs)` recebe só `input_ids` e `attention_mask`.

## Porque acontece

- **(a)** `parse_dot_edges` ([create_embeddings.py:113](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/create_embeddings.py#L113)) extrai só os pares de ids com uma expressão regular e ignora a etiqueta `DDG: x` / `CDG:`.
- **(b)** O `GraphConv` do DGL agrega das arestas de **entrada**. As arestas da AST vão de pai para filho, por isso a raiz, que é o nó que resume a função, nunca recebe informação dos filhos. No CFG, a informação só segue a ordem de execução. O DGCNN original foi proposto para grafos não dirigidos, onde a informação circula nos dois sentidos.
- **(c)** O que distingue o GraphCodeBERT do CodeBERT é receber, junto com o código, o grafo de data-flow das variáveis (`position_idx` e uma máscara de atenção própria). Aqui cada nó é um texto curto e isolado, por isso o modelo funciona como um codificador de texto genérico. O relatório justifica a escolha do GraphCodeBERT pela sua "data-flow awareness", que não é usada.

## Impacto nos resultados da tese

Nenhum destes pontos é um erro que invalide resultados. Mas limitam o que a DGCNN pode aprender e devem ser declarados. O ponto (c) afeta a forma como a contribuição do GraphCodeBERT é descrita na tese.

## Como resolver

- **(a)** Guardar o tipo de cada aresta (`edge_type` = 0 para CFG, 1 para CDG, 2 para DDG, …) e usar uma GNN relacional (`RelGraphConv` no DGL) ou features de aresta. Fundir arestas repetidas, ou guardar uma contagem.
- **(b)** Acrescentar as arestas inversas (`dgl.add_reverse_edges(g)`, idealmente com um tipo próprio) para que a informação suba e desça.
- **(c)** Uma de duas opções:
  1. codificar a função inteira com o GraphCodeBERT, com o data-flow, e atribuir a cada nó a média dos tokens das suas linhas (usando a linha guardada no `.dot`, ver 02);
  2. descrever na tese o uso como codificador de texto por nó.

## Limitações desta verificação

As contagens de (a) vêm do ficheiro de teste. O comportamento de (b) e (c) é do código e da biblioteca, por isso é geral.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 13_representacao_do_grafo\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
