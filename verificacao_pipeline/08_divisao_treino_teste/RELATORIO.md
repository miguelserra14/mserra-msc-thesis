# 08 — Treino e teste não são independentes

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: ALTA**

> A divisão 70/30 aleatória por amostra põe versões quase idênticas da mesma função dos dois lados. No PDG, 58% dos negativos e 23% dos positivos de teste têm uma "gémea" no treino. A avaliação não mede a capacidade de generalizar para código novo.

## Pergunta

As amostras do conjunto de teste são código que o modelo não viu no treino?

## Como o teste funciona

1. Reconstrói as amostras que chegam ao modelo: um `.pt` por caminho de `.dot`, com a primeira linha do CSV `-filtered`.
2. Aplica **exatamente** a divisão de [detect_vulnerabilities.py:640](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L640): `train_test_split(test_size=0.3, stratify=label, random_state=42+11·i)` para as 5 seeds.
3. Para cada amostra de teste, procura no treino:
   - a **mesma função**: mesmo ficheiro e nome, noutro commit;
   - uma **gémea quase idêntica**: além disso, a mesma assinatura e o mesmo número de linhas. É uma aproximação de "mesmo código"; o teste 09 confirma um caso byte a byte.
4. Demonstra a correção (divisão por grupo).

## Resultado

Média das 5 seeds:

| | AST | CFG | PDG |
|---|---|---|---|
| Amostras / pares (ficheiro, função) distintos | 101 476 / 45 129 | 101 455 / 45 191 | 101 620 / 45 123 |
| Positivos de teste com a mesma função no treino | 55,0% | 55,3% | 54,1% |
| Positivos de teste com gémea quase idêntica no treino | 23,8% | 23,6% | 23,3% |
| … dessa gémea, etiquetada vulnerável no treino | 3,2% | 2,7% | 3,0% |
| **Negativos de teste com gémea quase idêntica no treino** | **57,6%** | **57,3%** | **57,6%** |
| Com divisão por grupo (correção): gémeas no treino | 0% | 0% | 0% |

A divisão por grupo mantém a proporção de positivos (2,56%–2,66% no teste, contra 2,60% no total).

## Porque acontece

O dataset tem uma linha por (commit, ficheiro, função). Um mesmo ficheiro aparece em vários commits, um por cada CVE que lhe mexeu, e a maioria das funções não muda entre eles. Há 101 mil amostras, mas só 45 mil funções distintas. A divisão é feita amostra a amostra, e o DataFrame de treino nem guarda o commit ou o ficheiro (só `graphs` e `label`, [load_graphcodebert_embeddings.py:123](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/load_graphcodebert_embeddings.py#L123)). Por isso é impossível agrupar sem mudar o loader.

## Impacto nos resultados da tese

O efeito é diferente para as duas classes, e não se resume a "resultados otimistas":

- **Negativos:** mais de metade tem uma cópia no treino, quase sempre também negativa. Ficam mais fáceis de acertar. A accuracy acima de 95% vem sobretudo do desequilíbrio (97,4% são negativos), mas este efeito soma-se.
- **Positivos:** um quarto tem uma cópia quase idêntica no treino, e em 87% desses casos **nenhuma** das cópias está etiquetada como vulnerável. O modelo é treinado a dar 0 a esse mesmo código, o que puxa o recall para baixo (o mecanismo está no teste 09).
- Em qualquer dos casos, as métricas reportadas não estimam o desempenho em código que o modelo nunca viu, que é o cenário real de uso.

## Como resolver

1. O loader deve guardar os metadados que já estão nos `.pt`:

```python
# load_graphcodebert_embeddings.load_embeddings_from_directory
linhas.append({"graphs": g, "label": int(label), "commit": data_dict["commit"],
               "filepath": data_dict["filepath"], "function_name": data_dict["function_name"]})
df = pd.DataFrame(linhas)
```

2. Dividir **por grupo**, para que nenhuma função apareça dos dois lados (demonstrado no teste):

```python
from sklearn.model_selection import GroupShuffleSplit
grupos = run_df["filepath"] + "::" + run_df["function_name"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
i_tr, i_te = next(gss.split(run_df, run_df["label"], groups=grupos))
trainset_df, testset_df = run_df.iloc[i_tr], run_df.iloc[i_te]
```

   Para uma divisão estratificada e agrupada ao mesmo tempo, usar `StratifiedGroupKFold`.

3. A alternativa mais realista é uma **divisão temporal**: treinar com commits anteriores a uma data e testar com os posteriores, tal como uma ferramenta seria usada na prática. Mesmo assim convém retirar do teste as funções que já estão no treino.

## Limitações desta verificação

- Na pipeline real, a ordem das amostras é a de `os.listdir` sobre os `.pt`; aqui é a ordem do CSV. As percentagens por seed variam um pouco, mas as médias são as mesmas.
- "Gémea quase idêntica" é uma aproximação. Pode incluir funções com o mesmo tamanho mas código diferente, e excluir cópias reformatadas.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 08_divisao_treino_teste\teste.py
```

Não precisa da preparação. A saída completa está em [resultado.txt](resultado.txt).
