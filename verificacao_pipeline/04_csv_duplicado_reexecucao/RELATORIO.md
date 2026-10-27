# 04 — Voltar a correr a extração duplica o CSV de mapeamento

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: BAIXA**

> Cada reexecução acrescenta outra vez todas as linhas ao CSV. Os CSV reais têm entre 26% e 45% de linhas duplicadas. Sozinho não altera o dataset, mas é o que permite que linhas de extrações diferentes se misturem (teste 05).

## Pergunta

Correr `main.py` mais do que uma vez sobre o mesmo commit deixa o CSV de mapeamento igual?

## Como o teste funciona

A preparação (00) corre `map_single_commit_to_graph` duas vezes sobre o mesmo commit, como acontece quando `main.py` é relançado. O teste conta as linhas resultantes e mede as duplicações nos CSV reais do repositório.

## Resultado

No teste: **14 linhas, 7 únicas** em cada tipo. Mesmo assim saem só 6 `.pt` por tipo, porque o `create_embeddings.py` salta caminhos já processados.

Nos CSV reais (`output-data/`):

| Ficheiro | Linhas | Únicas | Duplicadas |
|---|---|---|---|
| functions-ast-linux.csv | 201 136 | 114 033 | 87 103 |
| functions-ast-linux-filtered.csv | 154 033 | 113 897 | 40 136 |
| functions-cfg-linux.csv | 202 631 | 111 076 | 91 555 |
| functions-cfg-linux-filtered.csv | 155 225 | 110 935 | 44 290 |
| functions-pdg-linux.csv | 202 248 | 114 283 | 87 965 |
| functions-pdg-linux-filtered.csv | 155 215 | 114 074 | 41 141 |

Duas observações:

- O `-filtered` é o original com **136/141/209 linhas únicas a menos e nenhuma a mais**. Não há no repositório nenhum script que o gere, e a diferença não corresponde a um filtro reconhecível.
- `map_functions_to_graph` escreve `functions-linux-{ast,cfg,pdg}.csv` (outra ordem no nome), que **nenhum script lê**.

## Porque acontece

[convert_dot_to_json.py:204](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/convert_dot_to_json.py#L204) abre o CSV em modo `'a'` (acrescentar) e nunca verifica se o commit já lá está. [main.py:65](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/main.py#L65) chama-o para todos os commits em cada execução, incluindo os que a extração salta por já estarem feitos.

## Impacto nos resultados da tese

Direto: nenhum, porque os `.pt` não se duplicam. Indireto: quando há linhas repetidas para o mesmo `.dot`, a primeira linha decide o nome e a etiqueta guardados no `.pt`; e linhas de extrações antigas ficam ao lado das novas sem forma de as distinguir. É exatamente o que o teste 05 encontra.

## Como resolver

Gerar o mapeamento **de raiz** a partir dos `.dot` que existem no disco, numa só passagem, em modo `'w'`, depois de terminada a extração:

```python
# no fim de main.py, em vez de map_single_commit_to_graph dentro do ciclo
for graph_type in graph_type_list:
    map_functions_to_graph(project, graph_type, csv_filename=filepath)   # já escreve em modo 'w'
```

Além disso:

- usar um só esquema de nomes (`functions-{tipo}-{projeto}.csv`) em `map_functions_to_graph` e `create_embeddings.py`;
- substituir o passo manual `-filtered` por código documentado (ou eliminá-lo);
- apagar os `functions-linux-*.csv` que não são usados.

## Limitações desta verificação

Não é possível saber, a partir dos CSV, quantas vezes cada commit foi processado nem por que ordem.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 04_csv_duplicado_reexecucao\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
