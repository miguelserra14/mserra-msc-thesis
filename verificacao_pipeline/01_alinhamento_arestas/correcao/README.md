# Correção do problema 01: alinhamento nós ↔ arestas

## 1. A correção

**O problema.** Em cada grafo, as features (o texto de cada nó, codificado pelo GraphCodeBERT) e as arestas usam ordens de nós diferentes:

- o loader empilha as features por ordem **numérica** dos ids do Joern;
- a extração numera as arestas por ordem de **texto** dos ids, e só conta os nós que têm arestas.

Os ids têm 11 e 12 dígitos, por isso as duas ordens quase nunca coincidem. A GNN recebe a forma certa do grafo, mas com os textos nos nós errados: 85 a 91% das arestas ligam fragmentos de código que não estão ligados no programa.

**O que se fez.** Uma só ordem de nós para features e arestas.

- **No loader**, que é onde o erro se manifesta, o `edge_index` gravado é renumerado para a ordem das features. Usa-se o `node_to_idx` que cada `.pt` já guarda e que diz exatamente como as arestas foram numeradas.
- **Na extração**, os `.pt` novos passam a sair numerados por ordem numérica, com todos os nós.
- **No treino**, a cache de grafos já construídos muda de nome, para que uma cache antiga não anule a correção sem aviso.
- **No `EmbeddingDataset`** ([embedding_dataset.py](../../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/embedding_dataset.py), a versão PyTorch Geometric do loader), que tinha o mesmo erro. Usa a mesma função e o mesmo interruptor do loader. Não entra no treino da tese, mas é a porta de entrada para as ferramentas de XAI do PyTorch Geometric. No modo corrigido, o `__init__` tira os grafos que não se podem alinhar, os mesmos que o loader DGL descarta. É preciso fazê-lo aí porque o `get()` não pode descartar um grafo: o seu `except` devolvia-o como um grafo vazio com etiqueta 0. Na amostra, o código original devolvia assim 72 falsos negativos.

**Porque é que se fez assim:**

- **Não é preciso regenerar o dataset.** A informação para alinhar já está em cada `.pt`. Basta usar o loader novo nos `.pt` da tese, sem voltar a correr o Joern nem o GraphCodeBERT, que levaram dias.
- **A correção mexe só nas arestas.** A matriz de features fica igual, valores e ordem. Assim, qualquer diferença nos resultados vem só do alinhamento, e o caminho AMP/VGG, que depende da ordem das linhas, não é afetado por outra coisa.
- **`.pt` antigos e novos dão o mesmo grafo.** Não há dois "corrigidos" diferentes, conforme a data de cada ficheiro.
- **O código da Constança fica intacto.** A correção só acrescenta linhas; nenhuma linha dela foi apagada ou alterada, comentários incluídos.
- **Dá para voltar ao original a qualquer momento.** A variável `USAR_CODIGO_ORIGINAL` em [interruptor.py](interruptor.py) troca entre o código dela (o baseline da tese) e o corrigido. Numa máquina sem esta pasta, faz-se o mesmo com `LEGACY_EDGE_ALIGNMENT=1` ou `=0`.

**Efeito secundário (único).** 2 de 2007 PDG da amostra têm uma aresta para um nó sem texto. O loader original carregava-os por acaso, mas com a estrutura errada. Agora são descartados, como os outros 221 PDG com o mesmo defeito, que já se perdiam ([falhas_loader](../testes_erro/falhas_loader/)).

## 2. Resultados

O relatório completo dos testes, com o que cada um prova e não prova, está em [RELATORIO_TESTES.md](RELATORIO_TESTES.md).

Os testes estão em [testar_correcao.py](testar_correcao.py) e passam todos ([resultado_correcao.txt](resultado_correcao.txt)). Seguem três regras, que é o que os torna fiáveis:

- **A verdade não vem do código testado.** As arestas verdadeiras são lidas do `.dot` do Joern com um parser independente.
- **Compara-se aresta a aresta.** Sabe-se que nó está em cada linha da matriz, por isso cada aresta é comparada como um par de nós do Joern, e não por contagens.
- **Há um controlo negativo.** O mesmo teste é corrido no código original e tem de falhar. Um teste que passasse sobre o código com o erro não provaria nada.

| Pergunta | Resultado | Porque importa |
|---|---|---|
| As arestas ficam certas? | **507 909/507 909** em 5984 grafos reais do kernel; 328/328 no ficheiro de teste (o original dá 50/328) | É a correção em si, em código real e em escala |
| A correção só muda as arestas? | Features, nós e ordem iguais em todos os grafos | Isola o efeito do alinhamento de qualquer outra mudança |
| `.pt` antigos e novos dão o mesmo grafo? | Idênticos em 6659/6659 | É o que permite não regenerar o dataset |
| O código original continua disponível e intacto? | Todas as linhas dela estão lá. No modo original, a extração e o loader reproduzem o código dela bit a bit (6659/6659). | Garante um baseline fiável para comparar com a tese |
| Os números de treino já obtidos valem para esta correção? | Grafos bit a bit iguais aos da condição "corrigido" em 5982/5984 | Na AST e no CFG, os números da amostra são os da correção real; no PDG são uma aproximação, por causa dos 2 grafos |
| O `EmbeddingDataset` (PyTorch Geometric) fica igual ao loader DGL? | Mesmos 5982 grafos, com `x` e `edge_index` iguais; 507 909/507 909 arestas certas. O original acerta 45 971 de 658 269 e devolve 72 grafos vazios com etiqueta 0. Os 7 mutantes são apanhados ([testes_adicionais.py](testes_adicionais.py), A1). | Quem usar o PyTorch Geometric (por exemplo, para XAI) recebe os mesmos grafos que o treino |
| Há outros caminhos que contornem a correção? | O único outro leitor de `.pt`, o `fusion_all_graphs.py`, reconstrói as arestas a partir do `.dot` e está alinhado (18/18; A5) | Nenhum ficheiro usa o `edge_index` gravado sem a correção |

**O que os números de treino dizem** (amostra de 5984 grafos reais, 3 seeds; ver [DETALHES.md](../DETALHES.md)):

- **XGBoost:** com a correção, sobe 0,05 a 0,10 de F1. No PDG, melhora nas 3 seeds.
- **SVM:** com e sem o erro dá o mesmo.
- **Sem arestas:** o modelo fica ao mesmo nível do corrigido, ou acima. A correção tira o ruído, mas nesta configuração a estrutura do grafo não acrescenta informação.

Isto muda a leitura da tese. Os números das tabelas não mudam de ordem de grandeza, mas **a tese não demonstra que a estrutura ajuda**, e isso não se resolve só com esta correção.

**O que ainda não está provado:**

- **Os `.pt` da tese.** A correção confia que o `node_to_idx` de cada `.pt` corresponde às suas arestas. Nesta máquina corresponde sempre, mas os `.pt` da tese não estão cá. Se algum tiver sido gerado por outra versão do código, seria realinhado mal sem dar erro.
- **O treino com o loader real.** Os grafos são iguais aos já treinados, mas não houve um treino novo a confirmá-lo.
- ~~**Arestas falsas no `.dot`.**~~ Fechado pelo A3 de [testes_adicionais.py](testes_adicionais.py): o Joern escapa as aspas do código como `&quot;`, por isso um padrão como `"123" -> "456"` dentro do código C nunca chega ao `.dot` como aresta (pelo menos nesta versão do Joern).

## 3. Próximos passos

1. **Verificar nos dados reais.** Na máquina com os `.pt` do Linux da tese, usar o loader corrigido e repetir a comparação com os `.dot` (o teste T3) numa amostra grande. É o único risco que não se consegue eliminar aqui. Se falhar, o loader não serve para esses ficheiros e é preciso regenerar o `edge_index` a partir dos `.dot`.
2. **Voltar a treinar as Tabelas 5.4–5.8 em três condições:** original (interruptor em `True`), corrigido e sem arestas, com as mesmas seeds e a mesma divisão. É esta comparação que diz quanto o erro custou e se a estrutura ajuda.
3. **Decidir o que a tese pode afirmar.** Se, mesmo corrigido, o modelo sem arestas continuar ao mesmo nível, a contribuição da GNN tem de ser reformulada. Isto deve ser discutido antes de investir mais no modelo.
4. **Estabilizar o treino da DGCNN.** O treino colapsa muitas vezes numa só classe, com ou sem o erro. Enquanto isso acontecer, as comparações entre condições têm muito ruído.
5. **Recuperar os grafos perdidos pelo loader** (os 223 PDG com nós sem texto, entre outros). É o problema seguinte, em [falhas_loader](../testes_erro/falhas_loader/).
