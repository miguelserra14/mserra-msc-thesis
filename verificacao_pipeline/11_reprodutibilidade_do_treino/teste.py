"""11 - O treino é reprodutível e as amostras são baralhadas entre épocas?

Verificação estática de detect_vulnerabilities.py (seeds e DataLoaders) e demonstração com a classe
ORIGINAL DGCNNEncoder de que, sem seed, duas inicializações dão pesos diferentes.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import CLASSIFICACAO, atribuicoes_de_topo, cabecalho, carregar_definicoes, ler_ast, secao, veredicto

import torch

DV = CLASSIFICACAO / "detect_vulnerabilities.py"
SEEDS = {"manual_seed", "seed", "manual_seed_all", "set_seed", "use_deterministic_algorithms"}


def main():
    cabecalho("11 - Reprodutibilidade do treino da DGCNN")
    arvore = ler_ast(DV)
    cfg = atribuicoes_de_topo(DV)

    secao("Seeds")
    chamadas = [n for n in ast.walk(arvore) if isinstance(n, ast.Call)]
    seeds = [(n.lineno, ast.unparse(n)) for n in chamadas
             if isinstance(n.func, ast.Attribute) and n.func.attr in SEEDS]
    print(f"chamadas que fixam seeds (torch/numpy/random/dgl): {seeds or 'nenhuma'}")
    print(f"SEED_LIST só é usada em train_test_split(random_state=seed)")

    secao("DataLoaders de treino")
    print(f"flags: USE_BOTH_WEIGHTING={cfg['USE_BOTH_WEIGHTING']}, USE_CLASS_WEIGHT={cfg['USE_CLASS_WEIGHT']}")
    loaders = []
    for n in sorted(chamadas, key=lambda c: c.lineno):
        if getattr(n.func, "id", "") == "DataLoader" and n.args and ast.unparse(n.args[0]) == "trainset":
            kws = {k.arg: ast.unparse(k.value) for k in n.keywords}
            baralha = kws.get("shuffle") == "True" or "sampler" in kws
            print(f"linha {n.lineno}: DataLoader(trainset, {', '.join(f'{k}={v}' for k, v in kws.items())}) -> baralha: {baralha}")
            loaders.append((n.lineno, baralha))
    # if USE_BOTH_WEIGHTING / elif not USE_CLASS_WEIGHT / else -> o ramo else é o último DataLoader do ficheiro
    ativo = loaders[-1]
    ramo_ativo = (not cfg["USE_BOTH_WEIGHTING"]) and cfg["USE_CLASS_WEIGHT"]
    print(f"com estas flags o ramo ativo é o último (linha {ativo[0]}): baralha = {ativo[1]}")
    print("train_test_split baralha uma vez; depois a ordem dos lotes é igual nas 30 épocas e nas 4 configurações")

    secao("Demonstração com a classe original DGCNNEncoder")
    g = carregar_definicoes(DV, {"DGCNNEncoder"}, {"dropout_rate": cfg["dropout_rate"]})
    a = g["DGCNNEncoder"](768, cfg["hidden_dimension"], k_sortpooling=cfg["k_sortpooling"])
    b = g["DGCNNEncoder"](768, cfg["hidden_dimension"], k_sortpooling=cfg["k_sortpooling"])
    iguais = torch.equal(a.conv1.weight, b.conv1.weight)
    print(f"duas instâncias sem seed têm os mesmos pesos iniciais? {iguais}")
    torch.manual_seed(0); c = g["DGCNNEncoder"](768, cfg["hidden_dimension"])
    torch.manual_seed(0); d = g["DGCNNEncoder"](768, cfg["hidden_dimension"])
    print(f"com torch.manual_seed(0) antes de cada uma? {torch.equal(c.conv1.weight, d.conv1.weight)}")

    problema = (not seeds) and ramo_ativo and not ativo[1]
    veredicto(problema, "MÉDIA",
              "sem seeds de torch/dgl, as 5 runs não são reprodutíveis (só a divisão é); e o DataLoader de treino ativo "
              "não baralha as amostras entre épocas. Não invalida resultados, mas impede repeti-los e pode piorar o treino.")


if __name__ == "__main__":
    main()
