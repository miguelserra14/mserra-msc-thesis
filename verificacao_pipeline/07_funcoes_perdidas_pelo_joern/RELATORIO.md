# 07 — O Joern perde funções, e perde mais as vulneráveis

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: ALTA**

> Ficam sem grafo 13,4% das funções vulneráveis contra 8,8% das não vulneráveis. Uma causa está provada com código real do kernel: como a pipeline analisa cada ficheiro sem os headers do Linux, anotações como `__user` e `__iomem` impedem o Joern de reconhecer a função. Que parte das perdas totais vem desta causa não está medida.

## Pergunta

Todas as funções do dataset chegam a ter grafo? E as que se perdem são uma amostra neutra?

## Como o teste funciona

1. **Dados reais:** para cada função de `linux-functions.csv`, verifica se tem caminho de `.dot` nos CSV de mapeamento e separa vulneráveis de não vulneráveis.
2. **Código real do kernel:** corre o Joern como a pipeline (um ficheiro isolado, sem headers) sobre dois ficheiros do Linux, nos commits exatos do dataset, guardados em [dados/linux_src/](../dados/linux_src/). Repete com `--define __user --define __iomem`.

## Resultado

**Funções sem grafo (dados reais):**

| Tipo | Vulneráveis sem grafo | Não vulneráveis sem grafo |
|---|---|---|
| AST | 407 / 3022 (13,5%) | 9554 / 107 793 (8,9%) |
| CFG | 396 / 3022 (13,1%) | 9221 / 107 793 (8,6%) |
| PDG | 406 / 3022 (13,4%) | 9481 / 107 793 (8,8%) |

No PDG, de 2458 ficheiros, 1096 têm algumas funções em falta e 27 não têm nenhuma. Ou seja, as perdas são sobretudo **parciais**: o Joern lê o ficheiro, mas salta funções.

**Código real:**

| Função | No dataset | Pipeline (sem headers) | Com `--define __user --define __iomem` |
|---|---|---|---|
| `ip_setsockopt` (net/ipv4/ip_sockglue.c @534afb90) | vulnerável, sem grafo | não exportada | **exportada** |
| `ip_getsockopt` (idem) | não vulnerável, sem grafo | não exportada | **exportada** |
| `iounmap` (arch/x86_64/mm/ioremap.c @1f5ee8da) | vulnerável, sem grafo | não exportada | **exportada** |
| `__ioremap` (idem) | não vulnerável, sem grafo | não exportada | **exportada** |

As assinaturas são `int ip_setsockopt(..., char __user *optval, int optlen)` e `void iounmap(volatile void __iomem *addr)`.

## Porque acontece

[graph_extraction.py:90](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/graph_extraction.py#L90) corre `joern-parse <ficheiro.c>` sem caminhos de include nem defines. Macros do kernel como `__user`, que marca ponteiros vindos do espaço do utilizador, ficam por definir. O parser de C não consegue ler a assinatura e o Joern descarta a função.

As funções com `__user` são precisamente as que copiam dados do utilizador para o kernel (`copy_from_user`, `*_setsockopt`, ioctls), uma das zonas mais ricas em CVEs. É plausível que isto explique parte da diferença entre 13,4% e 8,8%.

## Impacto nos resultados da tese

- O dataset perde 13% dos positivos, e não ao acaso: perde de forma desproporcionada uma família de funções muito relevante para segurança.
- O modelo nunca vê essas funções, e os resultados não dizem nada sobre elas.
- Algumas amostras negativas da mesma família também desaparecem; a distribuição muda nos dois lados.

## Como resolver

1. **Mínimo:** passar ao parser de C as anotações do kernel como defines. Chamar o frontend diretamente (o `c2cpg.sh` em `~/joern-cli`, que é o mesmo programa que o teste usa):

```python
DEFINES_KERNEL = ["__user", "__iomem", "__kernel", "__force", "__rcu", "__percpu", "__bitwise",
                  "__must_check", "__init", "__exit", "__cold", "__read_mostly", "asmlinkage", "notrace"]
cmd_parse = [os.path.join(JOERN_CLI_DIR, "c2cpg.sh"), filepath, "--output", TEMP_CPG_PATH]
for d in DEFINES_KERNEL:
    cmd_parse += ["--define", d]          # formato testado: "--define __user" (não "__user=")
```

   O teste provou apenas `__user` e `__iomem`. As restantes anotações da lista têm a mesma forma e é provável que causem o mesmo problema, mas **não foram testadas**.

2. **Mais completo:** passar os headers do kernel na versão do commit (`--include <linux>/include --include <linux>/arch/x86/include --define __KERNEL__`). É mais lento e pode trazer outros erros de parse.

3. **Em qualquer caso:** voltar a medir a taxa de funções sem grafo depois da mudança (a parte 1 deste teste faz isso), e reportar na tese quantas funções ficaram de fora e de que tipo.

## Limitações desta verificação

- A causa foi provada em 4 funções de 2 ficheiros. A parte das 406 funções vulneráveis perdidas que se deve a anotações não foi quantificada: o campo `Signature` do dataset não guarda as anotações, e verificar caso a caso exigiria descarregar milhares de ficheiros.
- Podem existir outras causas de perda (outras macros, sintaxe que o parser não aceita).

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 07_funcoes_perdidas_pelo_joern\teste.py
```

Não precisa da preparação nem de rede (os ficheiros do kernel estão em `dados/linux_src/`). Demora cerca de 30 s. A saída completa está em [resultado.txt](resultado.txt).
