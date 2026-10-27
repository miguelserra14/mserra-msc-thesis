# 00 — Preparação (não é um teste)

**Veredicto:** não se aplica. Este passo só cria os dados de que os testes 01–03, 13, 14 e 15 precisam.

## O que faz

Corre a pipeline **original** da Constança sobre um ficheiro C pequeno ([dados/vuln.c](../dados/vuln.c)), em vez do kernel Linux, e deixa o resultado em `_gerado/pipeline/`:

| Passo | Código usado | Saída |
|---|---|---|
| 1. Joern parse + export AST/CFG/PDG | Joern 4.0.649 (`c2cpg` + `joern-export`) | `output/linux/output-{tipo}-deadbeef…/drivers---test---vuln.c/N-{tipo}.dot` |
| 2. Texto dos nós | `graph_extraction.generate_txt_force` (original) | `N-{tipo}.dot.txt` |
| 3. Mapeamento função → `.dot` | `convert_dot_to_json.map_single_commit_to_graph` (original), **corrido 2 vezes** | `output-data/functions-{tipo}-linux.csv` |
| 4. CSV `-filtered` | passo manual reproduzido (não há script no repositório) | `functions-{tipo}-linux-filtered.csv`, coluna `CFG_filepath` |
| 5. Embeddings | `create_embeddings.py` (original) | `dataset_final_embeddings/{tipo}/*.pt` |

## Porque é que o ficheiro de teste tem esta forma

[dados/vuln.c](../dados/vuln.c) tem 6 funções escolhidas para provocar os casos difíceis:

- `copy_item`: chamada e condição escritas em várias linhas;
- `helper`: definida duas vezes, nos dois ramos de um `#ifdef` (uma marcada vulnerável de propósito no CSV);
- `log_msg`: string com `<BR/>`, `&`, aspas e `<tag>`;
- `compute`: variável não usada, `if` sem chaveta;
- `get_len`: função gerada por uma macro;
- `main`: chama as restantes.

## Alterações em relação ao código original

Só duas, ambas necessárias para correr fora do servidor da Constança:

1. **`GRAPH_TYPES`**: em cada cópia de `create_embeddings.py` muda-se só a linha `GRAPH_TYPES = ["pdg"]` para o tipo pretendido, exatamente como ela fazia à mão.
2. **Modelo GraphCodeBERT**: é substituído por [dados/stub_transformers.py](../dados/stub_transformers.py), que devolve vetores aleatórios de 768 dimensões, para não descarregar ~500 MB. Os testes que usam estes dados verificam o alinhamento, as etiquetas e o mapeamento, e nenhuma dessas coisas depende do valor dos vetores.

Os separadores `\` que o Windows produz no passo 3 são convertidos para `/` no passo 4, como nos CSV reais do repositório.

## Resultado

Ver [resultado.txt](resultado.txt). Saem 25 `.dot` por tipo (as 7 funções do ficheiro mais funções externas e operadores) e 6 `.pt` por tipo. As duas `helper` partilham um único `.pt`; isso é o problema do teste 03.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 00_preparacao\preparar.py
```

Demora cerca de 20 s. Apaga e recria `_gerado/pipeline/` em cada execução.
