# 05 — O CSV mistura linhas de extrações diferentes

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: MÉDIA**

> Nos CSV reais, 1766 ficheiros `.dot` da AST e 1768 do PDG estão ligados a duas funções com nomes diferentes. Como cada `.dot` contém uma só função, pelo menos uma linha de cada par está errada. Isto afeta cerca de 1,7% das amostras da AST e do PDG, com 68 casos de etiquetas contraditórias. O CFG quase não é afetado.

## Pergunta

Cada linha do CSV de mapeamento aponta para o `.dot` da função que nomeia?

## Como o teste funciona

1. **Prova que o problema não pode vir de uma só extração:**
   - exporta a AST de [dados/vuln.c](../dados/vuln.c) três vezes, a partir de dois parses independentes, e compara a numeração;
   - tenta exportar para uma pasta que já existe.
2. **Mede nos CSV reais** quantos `.dot` estão associados a mais do que um nome de função.

## Resultado

**Joern:**
- a numeração é determinística: nas 3 exportações, cada número corresponde sempre à mesma função (25 `.dot`);
- exportar para uma pasta existente falha com `error: Output directory ... already exists.` (código 1).

**CSV reais** (linhas únicas, só com caminho):

| Tipo | `.dot` com 2+ nomes | Desses, com etiquetas diferentes | Amostras do modelo |
|---|---|---|---|
| AST | 1766 | 68 | 101 476 |
| CFG | 1 | 0 | 101 455 |
| PDG | 1768 | 68 | 101 620 |

Exemplo: `iwl-agn-sta.c/18-ast.dot` aparece ligado a `iwl_send_lq_cmd` e a `is_lq_table_valid`.

## Porque acontece

O raciocínio tem três passos:

1. Um `.dot` tem um único `digraph "nome"`, e numa só execução o dicionário `nome → .dot` não liga dois nomes ao mesmo ficheiro.
2. A numeração do Joern é determinística e o `joern-export` não reescreve pastas existentes.
3. Logo, o CSV contém linhas de **pelo menos duas extrações que receberam entradas diferentes**. Isso só acontece com outro conteúdo do ficheiro (outro commit) ou outra versão do Joern. As pastas foram apagadas entre as duas, e as linhas antigas ficaram no CSV porque ele só cresce (teste 04).

A causa exata não é determinável sem os logs da extração. Uma hipótese compatível é um `git checkout` falhado que passou despercebido (teste 06). O facto de o CFG não ser afetado indica que a AST e o PDG foram extraídos mais do que uma vez.

## Impacto nos resultados da tese

O `create_embeddings.py` usa a **primeira** linha de cada caminho. Em cerca de 1,7% das amostras da AST e do PDG, o nome e possivelmente a etiqueta guardados no `.pt` podem não corresponder ao grafo. Em 68 casos as duas linhas têm etiquetas diferentes, por isso há no máximo 68 etiquetas trocadas, num universo de cerca de 2640 positivos. O efeito nas métricas é limitado, mas o dataset fica sem garantia de integridade.

## Como resolver

1. Regenerar o mapeamento de raiz a partir dos `.dot` que existem no disco (ver teste 04), com a correção por nome e linha (ver teste 03).
2. Validar cada linha antes de gerar as embeddings, rejeitando as que não batem certo:

```python
import html, re

def validar(row):
    with open(row["CFG_filepath"], encoding="utf-8", errors="ignore") as f:
        nome = re.match(r'digraph "(.*)" \{', f.readline())
    return nome is not None and html.unescape(nome.group(1)) == row["function_name"]

df = df[df.apply(validar, axis=1)]
```

3. Guardar a versão do Joern e o HEAD real de cada commit num ficheiro de log da extração.

## Limitações desta verificação

Sem os `.dot` reais, não é possível saber **qual** das duas linhas de cada par está certa, nem se há linhas antigas que apontem para um `.dot` reaproveitado sem colidir com nenhuma outra (esse caso não é detetável a partir do CSV).

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 05_mistura_de_extracoes\teste.py
```

Não precisa da preparação, mas precisa do Joern em `~/joern-cli`. Demora cerca de 15 s. A saída completa está em [resultado.txt](resultado.txt).

> Nota: na análise anterior, feita no chat, contei 1767/1769/2. A diferença de 1 é o caso das funções com o mesmo nome, que agora está separado no teste 03.
