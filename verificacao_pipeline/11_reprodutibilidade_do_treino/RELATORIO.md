# 11 — O treino não é reprodutível nem baralha os dados

**Veredicto: PROBLEMA CONFIRMADO · Gravidade: MÉDIA**

> O script de treino não fixa nenhuma seed de torch, numpy ou dgl (só a da divisão), por isso as 5 runs não podem ser repetidas. Com a configuração usada, o DataLoader de treino apresenta as amostras sempre pela mesma ordem nas 30 épocas.

## Pergunta

Correr outra vez o treino dá os mesmos resultados? As amostras são baralhadas entre épocas?

## Como o teste funciona

1. Procura em [detect_vulnerabilities.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py) chamadas que fixem seeds.
2. Lista os `DataLoader` de treino e determina qual está ativo com as flags do script.
3. Instancia duas vezes a classe **original** `DGCNNEncoder`, sem seed e com seed, e compara os pesos iniciais.

## Resultado

- Chamadas que fixam seeds: **nenhuma**. `SEED_LIST` só é passada a `train_test_split(random_state=seed)`.
- Flags: `USE_BOTH_WEIGHTING=False`, `USE_CLASS_WEIGHT=True`, o que ativa o último ramo:

| Linha | DataLoader | Baralha |
|---|---|---|
| 739 | `sampler=sampler` | sim |
| 756 | `sampler=sampler` | sim |
| **770 (ativo)** | só `batch_size` e `collate_fn` | **não** |

- Duas instâncias de `DGCNNEncoder` sem seed têm pesos iniciais diferentes; com `torch.manual_seed(0)` antes de cada uma, ficam iguais.

## Porque acontece

O script nunca chama `torch.manual_seed`, `np.random.seed`, `random.seed` nem `dgl.seed`. A inicialização das camadas, o dropout e a ordem interna do DGL mudam em cada execução. O DataLoader do ramo ativo ([linha 770](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/classification/detect_vulnerabilities.py#L770)) não tem `shuffle=True`: o `train_test_split` baralha uma vez, e a partir daí os lotes são sempre os mesmos e pela mesma ordem.

## Impacto nos resultados da tese

- Os números das Tabelas 5.3 e 5.8 não podem ser reproduzidos exatamente por terceiros, nem pela própria autora.
- A variação entre as 5 runs mistura o efeito da divisão com o da inicialização, sem forma de os separar.
- Sem baralhar, o gradiente vê sempre a mesma sequência de lotes. Isto pode piorar a convergência, mas não invalida os resultados.

## Como resolver

No início de cada run, dentro do ciclo `for run_idx in range(N_RUNS)`:

```python
import random, dgl
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dgl.seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)   # opcional; em GPU pode tornar o treino mais lento
```

E no ramo ativo:

```python
data_loader = DataLoader(trainset, batch_size=batch_size, collate_fn=collate, shuffle=True,
                         generator=torch.Generator().manual_seed(seed))
```

Também convém guardar no `hyperparameters.json` de cada run a seed e as versões de torch, dgl e transformers.

## Limitações desta verificação

Não mede quanto variam as métricas entre execuções com os dados reais.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 11_reprodutibilidade_do_treino\teste.py
```

Não precisa da preparação. A saída completa está em [resultado.txt](resultado.txt).
