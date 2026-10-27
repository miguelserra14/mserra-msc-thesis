# 09 — O mesmo código aparece com etiqueta 1 e com etiqueta 0

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: ALTA**

> Cerca de 27% dos positivos têm uma cópia quase idêntica etiquetada 0, e em quase 80% desses casos a cópia é de um commit **anterior**, ou seja, o mesmo código vulnerável antes de ser corrigido. Confirmado byte a byte num caso real. Não é um erro de código: resulta da forma como o dataset define os negativos.

## Pergunta

Pode o mesmo código aparecer no dataset com etiquetas diferentes?

## Como o teste funciona

1. Nas amostras que chegam ao modelo, conta os positivos com uma gémea quase idêntica (mesmo ficheiro, nome, assinatura e número de linhas) etiquetada 0, e verifica se essa gémea é de um commit anterior.
2. Para um caso concreto, `sctp_setsockopt_events` em `net/sctp/socket.c`, compara o código real do kernel nos dois commits. Os ficheiros, descarregados de `github.com/torvalds/linux`, estão em [dados/linux_src/](../dados/linux_src/).

## Resultado

| Tipo | Positivos | Com gémea quase idêntica etiquetada 0 | … de um commit anterior |
|---|---|---|---|
| AST | 2637 | 703 (26,7%) | 551 |
| CFG | 2633 | 703 (26,7%) | 550 |
| PDG | 2641 | 707 (26,8%) | 555 |

**Caso real:**
- em `eb1cf0f8` (2009-09-28), linhas 2045–2053: etiqueta **1**;
- em `6a55617e` (2008-08-21), linhas 2005–2013: etiqueta **0**;
- **código byte a byte igual** nas duas versões:

```c
static int sctp_setsockopt_events(struct sock *sk, char __user *optval,
					int optlen)
{
	if (optlen > sizeof(struct sctp_event_subscribe))
		return -EINVAL;
	if (copy_from_user(&sctp_sk(sk)->subscribe, optval, optlen))
		return -EFAULT;
	return 0;
}
```

No dataset, esta função aparece em 13 commits: 1 vez com etiqueta 1 e 12 vezes com etiqueta 0. Por causa do `__user` (teste 07), nenhuma das 13 versões tem grafo. O caso confirma o mecanismo, mas esta função em concreto não chega ao modelo; os 707 casos da tabela chegam.

## Porque acontece

O dataset de origem (o de João Henggeler, descrito em `A_Software_Vulnerability_Dataset_of_Large_Open_Source_C_C_Projects.pdf`) parte dos commits que corrigem CVEs. Para cada ficheiro alterado, marca como vulneráveis as funções que a correção mudou e como não vulneráveis **todas as outras funções do mesmo ficheiro**.

Quando um ficheiro é corrigido várias vezes ao longo dos anos, uma função vulnerável em 2009 já existia, igual, nas correções de 2008. Nesses commits não foi tocada, por isso aparece com 0. A etiqueta 0 quer dizer "não tocada por esta correção", não "não vulnerável".

## Impacto nos resultados da tese

- O modelo recebe exemplos contraditórios: o mesmo código com 1 e com 0. Com 707 de 2641 positivos nesta situação, o teto de recall possível baixa e o modelo é empurrado para prever 0.
- Juntamente com o teste 08, isto explica parte do recall muito baixo da Tabela 5.3 (F1 entre 0,00 e 0,15).
- A falha não é da Constança; é uma limitação do dataset que a tese deve discutir e tratar.

## Como resolver

Escolher uma (ou combinar):

1. **Retirar dos negativos as cópias de código vulnerável.** Calcular uma impressão digital do código de cada função (por exemplo, o hash do texto dos nós do `.dot` por ordem, ou do código fonte normalizado) e descartar as amostras com etiqueta 0 cuja impressão coincide com a de uma amostra com etiqueta 1. Esboço (não testado sobre os `.dot` reais):

```python
import hashlib, re
def impressao(dot_path):
    with open(dot_path, encoding="utf-8", errors="ignore") as f:
        codigo = sorted(re.findall(r'<BR/>([^<]*)> \]', f.read()))
    return hashlib.sha1("\n".join(codigo).encode()).hexdigest()

df["h"] = df.CFG_filepath.map(impressao)
vulneraveis = set(df[df.y == 1].h)
df = df[(df.y == 1) | ~df.h.isin(vulneraveis)]
```

2. **Usar como negativos as versões corrigidas** (colunas `Neutral Commit Hash` / `Neutral File Functions`, que a pipeline ignora). Isso dá pares vulnerável/corrigido, como no subconjunto emparelhado do PrimeVul.
3. **Propagar a etiqueta para trás:** marcar como vulnerável qualquer versão anterior com código idêntico ao de uma versão vulnerável, entre a introdução do erro e a correção (ideia do SZZ).

Em qualquer caso, deduplicar o dataset por impressão digital antes de dividir treino e teste (ver 08).

## Limitações desta verificação

A "gémea quase idêntica" é uma aproximação (mesma assinatura e tamanho). Só um caso foi confirmado com o código fonte; os 707 são uma estimativa.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 09_etiquetas_contraditorias\teste.py
```

Não precisa da preparação nem de rede. A saída completa está em [resultado.txt](resultado.txt).
