"""01 - Verifica, na máquina da tese, se os .pt podem ser usados com o loader corrigido.

Autónomo: só precisa de torch (não usa nada desta pasta). Copiar para a máquina com os .pt e correr:

  python verificar_pt_tese.py <pasta dataset_final_embeddings> [--dots <pasta extraction/>] [--max 0]

Para cada .pt verifica:
  E1  edge_index e node_to_idx são coerentes: node_to_idx é uma numeração 0..n-1 e o edge_index só usa esses índices;
  E2  a numeração é uma das duas que o código conhece: ordem de TEXTO só com nós com arestas (extração original) ou
      ordem NUMÉRICA com todos os nós (extração corrigida). Outra numeração = outra versão do código: não confiar;
  E3  (com --dots, se o .dot em dot_path existir) as arestas que o loader corrigido produz, como pares de ids do Joern,
      são exatamente as do .dot. É a verificação decisiva (o T3 de testar_correcao.py).
Sai com código 1 se algum .pt falhar E1, E2 ou E3. Os grafos que o loader corrigido descarta (aresta para um nó sem
texto) são contados à parte: não são falhas, o loader original também perdia quase todos.
"""
import argparse
import os
import re
import sys
from collections import Counter

import torch

RE_ARESTA = re.compile(r'^\s*"(\d+)"\s*->\s*"(\d+)"')


def chave(x):
    return (0, int(x), "") if x.isdigit() else (1, 0, x)


def arestas_do_dot(caminho):
    with open(caminho, encoding="utf-8", errors="replace") as f:
        return [(m.group(1), m.group(2)) for linha in f if (m := RE_ARESTA.match(linha))]


def verificar_pt(d, raiz_dots):
    """Devolve (estado, detalhe). estado: 'ok', 'ok_sem_dot', 'descartado', 'sem_arestas' ou 'FALHA_*'."""
    emb = d.get("embeddings") or {}
    ei = d.get("edge_index", torch.empty((2, 0), dtype=torch.long))
    n2i = d.get("node_to_idx") or {}
    if not emb:
        return "descartado", "sem nós"
    if ei.numel() == 0:
        return "sem_arestas", ""
    # E1
    if not n2i:
        return "FALHA_E1", "edge_index sem node_to_idx"
    if sorted(n2i.values()) != list(range(len(n2i))):
        return "FALHA_E1", "node_to_idx não é 0..n-1"
    if int(ei.min()) < 0 or int(ei.max()) >= len(n2i):
        return "FALHA_E1", "edge_index fora do node_to_idx"
    # E2
    por_indice = sorted(n2i, key=n2i.get)
    usados = set(ei.flatten().tolist())
    original = por_indice == sorted(n2i) and usados == set(range(len(n2i)))
    corrigida = por_indice == sorted(n2i, key=chave) and set(emb) <= set(n2i)
    if not (original or corrigida):
        return "FALHA_E2", "numeração desconhecida (nem a da extração original nem a da corrigida)"
    # o que o loader corrigido faz
    if any(n not in emb for n in n2i if n2i[n] in usados):
        return "descartado", "aresta para um nó sem texto"
    pares = Counter((por_indice[a], por_indice[b]) for a, b in zip(*ei.tolist()))
    versao = "indistinguível" if original and corrigida else ("original" if original else "corrigida")
    # E3
    dot = d.get("dot_path")
    if raiz_dots and dot and os.path.exists(os.path.join(raiz_dots, dot)):
        reais = Counter(arestas_do_dot(os.path.join(raiz_dots, dot)))
        if pares != reais:
            certas = sum((pares & reais).values())
            return "FALHA_E3", f"{certas}/{sum(reais.values())} arestas iguais ao .dot"
        return "ok", versao
    return "ok_sem_dot", versao


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta", help="pasta dataset_final_embeddings (com subpastas ast/cfg/pdg) ou uma pasta de .pt")
    ap.add_argument("--dots", help="pasta a que o dot_path dos .pt é relativo (normalmente extraction/)")
    ap.add_argument("--max", type=int, default=0, help="máximo de .pt por tipo (0 = todos)")
    a = ap.parse_args()
    pastas = [os.path.join(a.pasta, t) for t in ("ast", "cfg", "pdg") if os.path.isdir(os.path.join(a.pasta, t))] or [a.pasta]
    falhou = False
    for pasta in pastas:
        fich = sorted(f for f in os.listdir(pasta) if f.endswith(".pt"))
        if a.max:
            fich = fich[:a.max]
        estados, versoes, exemplos = Counter(), Counter(), {}
        for f in fich:
            try:
                d = torch.load(os.path.join(pasta, f), map_location="cpu")
                estado, det = verificar_pt(d, a.dots)
            except Exception as e:  # noqa: BLE001
                estado, det = "FALHA_leitura", str(e)[:80]
            estados[estado] += 1
            if estado in ("ok", "ok_sem_dot"):
                versoes[det] += 1
            if estado.startswith("FALHA"):
                exemplos.setdefault(estado, []).append(f"{f}: {det}")
        print(f"\n{pasta}: {len(fich)} .pt")
        for k, v in sorted(estados.items()):
            print(f"  {k:14} {v}")
        print(f"  extração: {dict(versoes)}")
        for k, ex in exemplos.items():
            for e in ex[:5]:
                print(f"  {k}: {e}")
        if estados["ok_sem_dot"] and not estados["ok"]:
            print("  AVISO: nenhum .dot encontrado; só E1/E2 verificados (use --dots para a verificação decisiva)")
        falhou |= any(k.startswith("FALHA") for k in estados)
    print("\nRESULTADO:", "HÁ .pt QUE NÃO SE PODEM USAR COM O LOADER CORRIGIDO" if falhou else "todos os .pt verificados estão bem")
    sys.exit(1 if falhou else 0)


if __name__ == "__main__":
    main()
