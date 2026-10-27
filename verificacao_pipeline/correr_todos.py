"""Corre a preparação e todos os testes; grava a saída de cada um em <pasta>/resultado.txt.

Uso (no ambiente conda que tem torch, dgl, pydot, pandas, sklearn, imblearn):
    C:\\Users\\migue\\miniconda3\\envs\\graphcodebert\\python.exe correr_todos.py            # tudo
    C:\\Users\\migue\\miniconda3\\envs\\graphcodebert\\python.exe correr_todos.py 01 08      # só estes testes
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def correr(script, destino):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore")
    inicio = time.time()
    r = subprocess.run([sys.executable, str(script)], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", env=env)
    saida = r.stdout + (("\n[stderr]\n" + r.stderr) if r.returncode != 0 and r.stderr else "")
    destino.write_text(saida, encoding="utf-8")
    return r.returncode, saida, time.time() - inicio


def main():
    filtro = sys.argv[1:]
    pastas = sorted(p for p in RAIZ.iterdir() if p.is_dir() and re.match(r"\d\d_", p.name))
    resumo = []
    for pasta in pastas:
        numero = pasta.name[:2]
        if filtro and numero not in filtro and numero != "00":
            continue
        if numero == "00" and filtro and (RAIZ / "_gerado" / "pipeline" / "dataset_final_embeddings").exists():
            continue                                    # preparação já feita; só se repete ao correr tudo
        script = pasta / ("preparar.py" if numero == "00" else "teste.py")
        print(f"> {pasta.name} ...", end=" ", flush=True)
        rc, saida, dt = correr(script, pasta / "resultado.txt")
        linha = next((l for l in saida.splitlines() if l.startswith("VEREDICTO:")), None)
        estado = linha.replace("VEREDICTO: ", "") if linha else ("concluída" if rc == 0 else f"ERRO (código {rc})")
        print(f"{estado}  [{dt:.0f}s]")
        resumo.append(f"{pasta.name:40} {estado}")
        if numero == "00" and rc != 0:
            print(saida[-3000:])
            sys.exit(1)

    if not filtro:                                      # o resumo só é reescrito quando se corre tudo
        (RAIZ / "_gerado").mkdir(exist_ok=True)
        (RAIZ / "_gerado" / "resumo.txt").write_text("\n".join(resumo) + "\n", encoding="utf-8")
        print("\nResumo gravado em _gerado/resumo.txt")
    print("Saída completa de cada teste em <pasta>/resultado.txt")


if __name__ == "__main__":
    main()
