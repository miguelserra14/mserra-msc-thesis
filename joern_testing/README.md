# joern_testing

Teste mínimo para perceber o que o Joern faz, sem o peso do pipeline do Linux.

## Como correr

```powershell
cd joern_testing
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

O Joern 4.0.649 está instalado em `C:\Users\migue\joern-cli` (fora do OneDrive, 1,75 GB comprimido). Precisa de Java, que já tinhas (versão 21).

## Ficheiros

| Ficheiro | O que é |
|---|---|
| `exemplo.c` | Uma função vulnerável (`strcpy` sem limite), a versão corrigida (`strncpy`) e um `main` que chama as duas |
| `run.ps1` | Corre os três passos abaixo |
| `consultas.sc` | Consultas ao grafo, na linguagem de query do Joern (Scala) |
| `out/` | Resultados, recriados em cada execução |
| `workspace/` | Cópia de trabalho que o Joern cria ao importar o CPG; pode ser apagada |

## Os três passos

**1. Parse: código → CPG.** O frontend de C (`c2cpg`) lê o código e grava `out/cpg.bin`, um único grafo com o ficheiro inteiro. Não compila nada, por isso funciona com código incompleto.

**2. Export: CPG → ficheiros `.dot`.** O `joern-export --repr <tipo>` escreve um `.dot` por função. Os tipos usados são `ast`, `cfg`, `pdg` e `cpg14` (os três juntos). É isto que o `graph_extraction.py` da Constança faz para cada ficheiro de cada commit.

**3. Consultas.** O `joern --script` carrega o CPG e permite interrogá-lo diretamente, sem passar pelos `.dot`.

## O que os resultados mostram

**Saem 14 grafos, mas só 3 interessam.** Cada pasta de `out/` tem 14 `.dot`: as 3 funções do ficheiro, mais `strcpy`, `printf`, `strncpy`, os operadores (`<operator>.assignment`, etc.), a macro `BUF_SIZE` e o `<global>`. Num dataset é preciso filtrar estes grafos extra.

**Formato de um nó.** Cada nó do `.dot` tem um id, o tipo, a linha e o código:

```
"30064771075" [label = <strcpy, 9<BR/>strcpy(buf, input)> ]
```

O número a seguir ao tipo (`9`) é a linha no ficheiro original. É esta informação que permite passar de relevância por nó para relevância por linha. O `generate_txt_force` da Constança guarda só o código e deita fora a linha.

**Tamanho dos grafos.** O PDG da função vulnerável tem 9 nós e 22 arestas; o da corrigida tem 16 nós e 36 arestas. O par vulnerável/corrigido difere em poucos nós, o que dá uma ideia de quão fino tem de ser o sinal que o modelo aprende.

**As macros aparecem como chamadas.** `BUF_SIZE` surge como um nó de chamada, não como o literal 16.

**O fluxo de dados atravessa funções.** A consulta 5 pergunta se `argv` chega ao segundo argumento do `strcpy`, e o Joern devolve o caminho completo:

```
main(int argc, char **argv)     linha 21   main
copy_input(argv[1])             linha 25   main
copy_input(const char *input)   linha 7    copy_input
strcpy(buf, input)              linha 9    copy_input
```

Os `.dot` exportados são por função e não têm esta ligação. A parte interprocedural da tese tem de sair de consultas ao CPG (call graph e data flow), não do `joern-export`.

## Problema encontrado

O `joern-parse.bat` falha nesta versão em Windows com `-cp requires class path specification`. É um defeito do launcher do `c2cpg`, que passa uma classpath vazia ao Java. O `run.ps1` contorna-o chamando o frontend diretamente:

```powershell
java -jar $HOME\joern-cli\frontends\c2cpg\lib\io.joern.c2cpg-4.0.649-launcher.jar exemplo.c --output out\cpg.bin
```

O `joern-export.bat` e o `joern.bat` funcionam normalmente. Em Linux (onde corre o pipeline da Constança) o `joern-parse` não deve ter este problema, mas não testei.

## Experiências a seguir

- Abrir um `.dot` em https://dreampuf.github.io/GraphvizOnline para ver o grafo desenhado.
- Comparar `out/pdg/0-pdg.dot` (vulnerável) com `out/pdg/2-pdg.dot` (corrigida).
- Correr `C:\Users\migue\joern-cli\joern.bat` sem argumentos para abrir a consola interativa, fazer `importCpg("out/cpg.bin")` e experimentar consultas como `cpg.method.name.l` ou `cpg.call.name("strcpy").l`.
- Alterar `exemplo.c` (por exemplo, acrescentar um `if (strlen(input) < BUF_SIZE)`) e ver como mudam o CFG e o PDG.
