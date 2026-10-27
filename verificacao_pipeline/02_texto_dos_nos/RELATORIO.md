# 02 — Identificadores e literais recebem o texto errado

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: MÉDIA**

> Os nós IDENTIFIER e LITERAL são enviados ao GraphCodeBERT com o texto da expressão onde estão, não com o seu próprio código. Afeta sobretudo a AST (43 de 144 nós no teste). O código multilinha e strings com `<BR/>` **não** são problema.

## Pergunta

O texto que [`generate_txt_force`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/graph_extraction.py#L22) escreve em cada `.txt`, e que vai ser codificado pelo GraphCodeBERT, corresponde ao código do nó?

## Como o teste funciona

Para cada nó das 6 funções de [dados/vuln.c](../dados/vuln.c), compara a linha do `.txt` gerado pelo código original com os campos da etiqueta no `.dot`. Verifica também se todas as linhas do `.txt` podem ser lidas pelo `create_embeddings.py`, e aplica a correção proposta.

## Resultado

| Tipo | Nós com texto errado |
|---|---|
| AST | 43 de 144 |
| CFG | 0 de 68 |
| PDG | 4 de 68 |

Exemplos da AST de `compute`:

| Tipo | Código do nó | Texto no `.txt` | Com a correção |
|---|---|---|---|
| IDENTIFIER | `a` | `a + b` | `a` |
| IDENTIFIER | `b` | `a + b` | `b` |
| LITERAL | `5` | `unused = 5` | `5` |
| LITERAL | `10` | `r > 10` | `10` |

Outras observações:

- Na AST, **54 nós têm exatamente o mesmo texto que outro nó do mesmo grafo**, e portanto a mesma embedding (o GraphCodeBERT é determinístico).
- O tipo do nó (METHOD, CALL, IDENTIFIER…) e o número da linha são descartados. A linha é precisamente o que permitiria passar de relevância por nó para relevância por linha.
- O Joern trunca o texto de nós longos com `...`; por exemplo, o nó do `memcpy` fica `memcpy(dst->data, src->data, ...` e perde o argumento do tamanho. Isto é uma limitação do Joern, não do código.

**Sem problema (verificado):** nenhuma linha dos `.txt` fica ilegível. O Joern junta o código multilinha numa só linha, e um `<BR/>` dentro de uma string do código fonte chega escapado ao `.dot`, por isso a divisão por `<BR/>` não o parte.

## Porque acontece

O Joern escreve as etiquetas como `TIPO, linha<BR/>código` e, em IDENTIFIER e LITERAL, acrescenta um terceiro campo com o código da expressão-mãe:

```
IDENTIFIER, 43<BR/>a<BR/>a + b
LITERAL, 42<BR/>5<BR/>unused = 5
CONTROL_STRUCTURE, 44<BR/>IF<BR/>if (r > 10) r = 10;
```

[graph_extraction.py:43](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/graph_extraction.py#L43) fica sempre com o último campo (`split("<BR/>")[-1]`). Esse campo é o código certo para CONTROL_STRUCTURE, mas não para IDENTIFIER nem LITERAL.

## Impacto nos resultados da tese

No ficheiro de teste, 30% dos nós da AST são identificadores e literais. Ficam com a embedding da expressão-mãe, o que duplica informação e apaga o que distingue um operando de outro (o tamanho de um buffer, o nome da variável). No CFG não há efeito, e no PDG só nos literais. Isto contribui para a AST ser a representação com menos informação útil.

## Como resolver

Em `generate_txt_force` (e na cópia em `extract_mising.py`), escolher o campo conforme o tipo do nó e guardar também o tipo e a linha:

```python
campos = clean_label.split("<BR/>")
tipo, _, linha = campos[0].partition(", ")
if tipo in ("IDENTIFIER", "LITERAL") and len(campos) >= 3:
    codigo = campos[1]            # código do próprio nó
else:
    codigo = campos[-1]
final_code = html.unescape(codigo).strip()
f.write(f"{node_id} {final_code}\n")
# opcional, para relevância por linha: guardar (node_id, tipo, linha) num ficheiro .meta ao lado
```

Depois é preciso voltar a gerar os `.txt` e as embeddings da AST e do PDG. Os `.dot` não mudam.

## Limitações desta verificação

As contagens vêm do ficheiro de teste. A proporção de nós afetados nos grafos reais do Linux não foi medida (faltam os `.dot` reais), mas o formato das etiquetas é o mesmo, como mostra o apêndice do relatório da Constança (`LITERAL, 57<BR/>0<BR/>__data[0]`).

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 02_texto_dos_nos\teste.py
```

Precisa da preparação (00). A saída completa está em [resultado.txt](resultado.txt).
