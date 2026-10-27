# Verificação da pipeline GraphCodeBERT + DGCNN (Constança Meneses)

Testes executáveis que verificam, um a um, os problemas encontrados na pipeline de [`uc.-msc-constanca-graphcodebert-vulnerability-detection`](../uc.-msc-constanca-graphcodebert-vulnerability-detection/). Cada pasta tem:

- o teste (`teste.py`);
- a saída da última execução (`resultado.txt`);
- um relatório (`RELATORIO.md`) que diz se há problema, qual, com que gravidade e como resolver.

Os testes **leem** o repositório da Constança e **nunca o alteram**. Sempre que possível correm o código original (importado tal como está) sobre dados reais: os CSV do Linux, código do kernel nos commits exatos do dataset e o texto do relatório final.

## Resumo

| # | Teste | Veredicto | Gravidade | Numa frase |
|---|---|---|---|---|
| [01](01_alinhamento_arestas/RELATORIO.md) | Alinhamento nós ↔ arestas | Confirmado | 🔴 **CRÍTICA** | Só 15% das arestas ligam os nós certos; o conteúdo dos nós fica trocado. A correção recupera 100%. |
| [07](07_funcoes_perdidas_pelo_joern/RELATORIO.md) | Funções perdidas pelo Joern | Confirmado | 🔴 ALTA | Perdem-se 13,4% das funções vulneráveis contra 8,8% das restantes; `__user`/`__iomem` sem headers é uma causa provada. |
| [08](08_divisao_treino_teste/RELATORIO.md) | Divisão treino/teste | Confirmado | 🔴 ALTA | 58% dos negativos e 23% dos positivos de teste têm uma gémea quase idêntica no treino. |
| [09](09_etiquetas_contraditorias/RELATORIO.md) | Etiquetas contraditórias | Confirmado | 🔴 ALTA | 27% dos positivos têm uma cópia etiquetada 0; num caso real o código é byte a byte igual. |
| [10](10_selecao_no_conjunto_de_teste/RELATORIO.md) | Seleção no conjunto de teste | Confirmado | 🔴 ALTA | Optuna e "melhor run" escolhidos pelo F1 no teste; não há validação. |
| [02](02_texto_dos_nos/RELATORIO.md) | Texto dos nós | Confirmado | 🟠 MÉDIA | IDENTIFIER/LITERAL recebem o texto da expressão-mãe (`a` → `a + b`). |
| [05](05_mistura_de_extracoes/RELATORIO.md) | Mistura de extrações no CSV | Confirmado | 🟠 MÉDIA | 1766 `.dot` (AST) e 1768 (PDG) ligados a duas funções; 68 com etiquetas diferentes. |
| [06](06_git_checkout_silencioso/RELATORIO.md) | `git checkout` silencioso | Risco | 🟠 MÉDIA | Um checkout falhado não pára a extração; não se sabe se aconteceu. |
| [11](11_reprodutibilidade_do_treino/RELATORIO.md) | Reprodutibilidade do treino | Confirmado | 🟠 MÉDIA | Sem seeds de torch/dgl; o DataLoader ativo não baralha. |
| [12](12_svm_sobre_embeddings_de_treino/RELATORIO.md) | SVM sobre embeddings de treino | Confirmado | 🟠 MÉDIA | O SVM aprende com embeddings memorizadas pelo encoder (100% vs 48% com etiquetas aleatórias). |
| [03](03_funcoes_mesmo_nome/RELATORIO.md) | Funções com o mesmo nome | Confirmado | 🟡 BAIXA | Partilham `.dot` e etiqueta; nos dados do Linux só 61 funções e 1 conflito. |
| [04](04_csv_duplicado_reexecucao/RELATORIO.md) | CSV duplicado ao reexecutar | Confirmado | 🟡 BAIXA | Cada execução acrescenta as linhas outra vez (26–45% de duplicados). |
| [13](13_representacao_do_grafo/RELATORIO.md) | Representação do grafo | Limitação | 🟡 BAIXA | Perde-se o tipo de aresta do PDG; a AST só propaga da raiz para as folhas; GraphCodeBERT sem data-flow. |
| [14](14_arquitetura_dgcnn/RELATORIO.md) | Arquitetura DGCNN | Desvio | 🟡 BAIXA | Só a última camada GraphConv; o adaptador VGG duplica 96 de 128 colunas. |
| [15](15_verificacoes_sem_problema/RELATORIO.md) | Verificações sem problema | Sem problema | ⚪ NENHUMA | 15/15 OK: etiquetas, `.pt`, SMOTETomek, PCA, pesos de classe. |
| [00](00_preparacao/RELATORIO.md) | Preparação | — | — | Corre a pipeline original sobre [dados/vuln.c](dados/vuln.c) para os testes 01–03, 13–15. |

### Escala de gravidade

- **CRÍTICA**: o modelo não recebe o que se julga que recebe; os resultados não medem o que dizem medir.
- **ALTA**: enviesa as métricas reportadas ou o dataset de forma mensurável.
- **MÉDIA**: afeta parte dos dados, a reprodutibilidade, ou é um risco latente.
- **BAIXA**: desenho, eficiência ou desvio em relação ao descrito; não invalida resultados.

## Como os problemas se relacionam

```
extração        07 funções perdidas ─┐
                06 checkout? ──► 05 mistura de extrações ◄── 04 CSV só cresce
                03 nomes repetidos ──┘
texto dos nós   02 identificadores/literais
grafo           01 alinhamento (CRÍTICO) · 13 tipos de aresta / sentido
dataset         09 etiquetas contraditórias ──► 08 gémeas entre treino e teste
treino          11 seeds/shuffle · 14 arquitetura
avaliação       10 seleção no teste · 12 SVM in-sample
```

## Ordem sugerida para corrigir

1. **01** (alinhamento). Não exige recalcular embeddings: basta reescrever o `edge_index` dos `.pt`.
2. **07 + 02 + 03 + 04 + 05 + 06**, numa nova extração: defines do kernel no parser, texto correto dos nós, mapeamento por nome e linha, CSV gerado de raiz e validado, checkout verificado.
3. **09 + 08** (dataset): deduplicar por código, decidir o que fazer com os negativos (versões corrigidas?), dividir por grupo ou por data.
4. **10 + 11 + 12** (avaliação): conjunto de validação, seeds, média ± desvio, SVM com embeddings fora da amostra.
5. **14 + 13**, se houver tempo, ou descrevê-los como limitações.

## Como correr

Requisitos (já instalados nesta máquina):

- ambiente conda `graphcodebert` (torch, dgl, pydot, pandas, scikit-learn, imbalanced-learn, GitPython);
- Joern em `~/joern-cli` e Java 21;
- o repositório da Constança em `../uc.-msc-constanca-graphcodebert-vulnerability-detection`, com os CSV de `extraction/output-data/` e `extraction/function-data/`.

```powershell
cd "C:\Users\migue\OneDrive\Documentos\trabalho\Curso\5º Ano\Tese\verificacao_pipeline"
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe correr_todos.py          # tudo (~4 min)
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe correr_todos.py 01 08    # só alguns
```

O `correr_todos.py` grava a saída de cada teste no seu `resultado.txt` e, quando corre tudo, um resumo em `_gerado/resumo.txt`. Os testes 01–03, 13, 14 e 15 precisam da preparação (00); o `correr_todos.py` trata disso.

## Estrutura

```
verificacao_pipeline/
├── README.md               este ficheiro
├── comum.py                caminhos, chamadas ao Joern, leitura de .dot e CSV, veredicto
├── correr_todos.py         corre tudo e grava os resultados
├── dados/
│   ├── vuln.c              ficheiro C de teste (casos difíceis)
│   ├── stub_transformers.py  substituto do GraphCodeBERT (vetores aleatórios; ver 00)
│   ├── relatorio_constanca.txt  texto do relatório final (pdftotext)
│   └── linux_src/          4 ficheiros do kernel nos commits do dataset (github.com/torvalds/linux)
├── 00_preparacao/ … 15_verificacoes_sem_problema/   teste.py · resultado.txt · RELATORIO.md
└── _gerado/                saídas intermédias (pode ser apagado; é recriado)
```

## Limites desta verificação

- Os `.dot`, `.txt` e `.pt` reais do Linux **não estão nesta máquina**. Os testes que precisam de grafos usam o ficheiro de teste; os que precisam de escala usam os CSV reais. Nenhum teste repete o treino com os dados reais.
- O GraphCodeBERT é substituído por vetores aleatórios na preparação. Nenhum teste depende do valor dos vetores.
- O código legado das features manuais não foi auditado linha a linha, porque não entra nos resultados com GraphCodeBERT (secção 5.2 do relatório).
- Não é possível garantir que não existem outros problemas. Os ficheiros que entram na pipeline foram lidos na íntegra, e cada afirmação dos relatórios tem um teste por trás.
