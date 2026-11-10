"""Resume os resultados de treinar.py: com o erro vs. corrigido vs. sem arestas, por tipo e classificador.

Para cada métrica mostra a média entre seeds e a diferença emparelhada (corrigido - erro), seed a seed.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTADOS = Path(__file__).resolve().parents[3] / "_gerado" / "impacto_tese" / "resultados"
CONDICOES = ["erro", "corrigido", "sem_arestas"]
METRICAS = ["f1", "recall", "precisao", "roc_auc", "pr_auc", "f1_aj_2.6"]


def main():
    dados = defaultdict(dict)
    for p in sorted(RESULTADOS.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        dados[(r["tipo"], r["seed"])][r["condicao"]] = r
    if not dados:
        print("sem resultados")
        return
    exemplo = next(iter(next(iter(dados.values())).values()))
    print(f"amostra: tipo -> grafos (vulneráveis), perdidos no loader")
    for tipo in ("pdg", "cfg", "ast"):
        rs = [r for (t, _), c in dados.items() if t == tipo for r in c.values()]
        if rs:
            print(f"  {tipo}: {rs[0]['n']} ({rs[0]['n_vuln']}), teste {rs[0]['n_teste']} ({rs[0]['n_vuln_teste']}), perdidos {rs[0]['perdidos_loader']}")

    for clf in ("svm", "xgb", "conv1d"):
        print(f"\n=== {clf.upper()} {'(T9 -> SMOTETomek -> ' + clf.upper() + ', como na Tabela 5.7)' if clf != 'conv1d' else '(classificador da própria DGCNN)'}")
        print(f"{'tipo':4} {'métrica':10} | {'com o erro':>12} {'corrigido':>12} {'sem arestas':>12} | {'corrigido - erro, por seed':>30}")
        for tipo in ("pdg", "cfg", "ast"):
            seeds = sorted(s for (t, s), c in dados.items() if t == tipo and all(k in c for k in ("erro", "corrigido")))
            if not seeds:
                continue
            for m in METRICAS:
                cols = []
                for cond in CONDICOES:
                    v = [dados[(tipo, s)][cond][clf][m] for s in seeds if cond in dados[(tipo, s)]]
                    cols.append(f"{st.mean(v):.3f}" + (f"±{st.stdev(v):.3f}" if len(v) > 1 else "") if v else "-")
                dif = [dados[(tipo, s)]["corrigido"][clf][m] - dados[(tipo, s)]["erro"][clf][m] for s in seeds]
                print(f"{tipo:4} {m:10} | {cols[0]:>12} {cols[1]:>12} {cols[2]:>12} | "
                      + " ".join(f"{d:+.3f}" for d in dif) + f"  (seeds {seeds})")
    print("\nprecisão/F1 'aj_2.6': recalculados para a prevalência da tese (2,6% vulneráveis) a partir do recall e da taxa"
          " de falsos positivos; a amostra tem mais vulneráveis do que o dataset completo.")
    colapsos = [(t, c, s) for (t, s), cs in dados.items() for c, r in cs.items()
                if r["conv1d"]["fracao_positivos_previstos"] in (0.0, 1.0)]
    print(f"treinos em que a DGCNN prevê uma só classe no teste: {len(colapsos)} de {sum(len(c) for c in dados.values())}")


if __name__ == "__main__":
    main()
