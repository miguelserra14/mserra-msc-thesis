"""Corre treinar.py para todas as combinações, N de cada vez. Retomável: salta o que já tem resultado.

Ordem: por seed, depois tipo, depois condição. Assim, se for interrompido, os resultados já obtidos estão
emparelhados (as 3 condições de um tipo e de uma seed ficam completas antes de passar à seguinte).
"""
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RESULTADOS = AQUI.parents[2] / "_gerado" / "impacto_tese" / "resultados"
SEEDS = [42, 53, 64]                     # as 3 primeiras de SEED_LIST em detect_vulnerabilities.py
TIPOS = ["pdg", "cfg", "ast"]
CONDICOES = ["erro", "corrigido", "sem_arestas"]
PARALELO = int(sys.argv[1]) if len(sys.argv) > 1 else 2


def main():
    tarefas = [(t, c, s) for s in SEEDS for t in TIPOS for c in CONDICOES
               if not (RESULTADOS / f"{t}_{c}_{s}.json").exists()]
    print(f"{len(tarefas)} treinos por fazer, {PARALELO} em paralelo", flush=True)
    a_correr = []
    t0 = time.time()
    while tarefas or a_correr:
        while tarefas and len(a_correr) < PARALELO:
            t, c, s = tarefas.pop(0)
            log = open(RESULTADOS.parent / f"treino_{t}_{c}_{s}.log", "w", encoding="utf-8")
            p = subprocess.Popen([sys.executable, "-X", "utf8", "-u", str(AQUI / "treinar.py"), "--tipo", t, "--condicao", c,
                                  "--seed", str(s), "--threads", "3"], stdout=log, stderr=subprocess.STDOUT)
            a_correr.append((p, (t, c, s), log, time.time()))
            print(f"[{(time.time() - t0) / 60:6.1f} min] início {t} {c} {s}", flush=True)
        time.sleep(10)
        for item in list(a_correr):
            p, (t, c, s), log, ti = item
            if p.poll() is not None:
                log.close()
                a_correr.remove(item)
                estado = "ok" if p.returncode == 0 else f"ERRO {p.returncode}"
                print(f"[{(time.time() - t0) / 60:6.1f} min] fim {t} {c} {s}: {estado} ({(time.time() - ti) / 60:.0f} min)", flush=True)
    print("FIM", flush=True)


if __name__ == "__main__":
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    main()
