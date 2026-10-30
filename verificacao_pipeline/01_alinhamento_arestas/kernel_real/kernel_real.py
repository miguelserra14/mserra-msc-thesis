"""Mede o erro do teste 01 em código real do kernel (dados/linux_src), com a pipeline original."""
import contextlib, io, os, re, statistics as st, sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aprendizagem"))
import teste_aprofundado as A
from comum import DADOS, EXTRACAO, correr_joern, ler_dot, campos_label
import torch

sys.path.insert(0, str(EXTRACAO))
from graph_extraction import generate_txt_force

linhas = []
for fonte in sorted((DADOS / "linux_src").glob("*.c")):
    print("Joern:", fonte.name, flush=True)
    pastas = correr_joern(fonte, ["ast", "cfg", "pdg"])
    for tipo, pasta in pastas.items():
        for dot in pasta.glob("*.dot"):
            nome, labels, arestas = ler_dot(dot)
            # só funções definidas no ficheiro: o nó METHOD tem número de linha
            if not any(re.match(r"METHOD, \d+", campos_label(l)[0]) for l in labels.values()) or nome.startswith(("<", "&lt;")):
                continue
            os.chdir(dot.parent)
            with contextlib.redirect_stdout(io.StringIO()):
                generate_txt_force(dot.name)
            emb, lixo = {}, 0
            txt = dot.with_name(dot.name + ".txt")
            if not txt.exists():
                linhas.append(dict(f=fonte.name, tipo=tipo, nome=nome, estado="sem_txt"))
                continue
            with open(txt, encoding="utf-8", errors="ignore") as f:
                for l in f:                                     # igual a create_embeddings.py
                    p = l.strip().split(" ", 1)
                    if len(p) == 2:
                        emb[p[0]] = torch.zeros(4)
                        lixo += not p[0].isdigit()
            ei, n2i = A.parse_dot_edges(str(dot))
            r = dict(f=fonte.name, tipo=tipo, nome=nome, nos_dot=len(labels), arestas=len(arestas), nos=len(emb), lixo=lixo,
                     isolados=len(set(emb) - set(n2i)), em_falta=len(set(n2i) - set(emb)))
            if not emb:
                r["estado"] = "sem_nos"; linhas.append(r); continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                g = A.create_dgl_graph_from_embedding(dict(embeddings=emb, edge_index=ei, node_to_idx=n2i))
            r["falta_txt"] = [campos_label(labels[n]) for n in set(n2i) - set(emb) if n in labels][:3]
            r["falta_sem_label"] = sum(n not in labels for n in set(n2i) - set(emb))
            if g is None:
                r["estado"] = "loader_falha"; r["erro"] = buf.getvalue().strip()[:160]; linhas.append(r); continue
            ordem = sorted(emb, key=int)
            papel = [n for n, _ in sorted(n2i.items(), key=lambda kv: kv[1])]
            verdade = {(a, b) for a, b, _ in arestas}
            s, d = g.edges()
            vistas = [(ordem[i], ordem[j]) for i, j in zip(s.tolist(), d.tolist())]
            comuns = [(i, a, b) for i, (a, b) in enumerate(zip(papel, ordem))]
            m = sum(len(n) == 12 for n in papel)
            r.update(estado="ok", certas=sum(e in verdade for e in vistas), total=len(vistas),
                     nos_trocados=sum(a != b for _, a, b in comuns), nos_comuns=len(comuns), m=m,
                     rotacao=len(papel) == len(ordem) and ordem == [papel[(i + m) % len(papel)] for i in range(len(papel))],
                     digitos=tuple(sorted({len(n) for n in n2i})),
                     outro_tipo=sum((int(a) >> 32) != (int(b) >> 32) for _, a, b in comuns))
            linhas.append(r)

os.chdir(RAIZ)
print("\nfunções por ficheiro:", Counter((r["f"], r["tipo"]) for r in linhas))
print("estados:", Counter((r["tipo"], r["estado"]) for r in linhas))
for tipo in ("ast", "cfg", "pdg"):
    o = [r for r in linhas if r["tipo"] == tipo and r["estado"] == "ok" and r["total"] > 0]
    if not o:
        continue
    pct = sorted(r["certas"] / r["total"] for r in o)
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    nos = [r["nos"] for r in o]
    print(f"\n{tipo}: {len(o)} funções com arestas")
    print(f"  nós: mediana {st.median(nos)}, p10 {q(nos,.1)}, p90 {q(nos,.9)}, máx {max(nos)}; >32 nós: {sum(n > 32 for n in nos)} ({sum(n > 32 for n in nos) / len(o) * 100:.0f}%)")
    print(f"  arestas: mediana {st.median(r['total'] for r in o)}")
    print(f"  arestas certas: agregado {sum(r['certas'] for r in o)}/{sum(r['total'] for r in o)} = {sum(r['certas'] for r in o) / sum(r['total'] for r in o) * 100:.1f}%; "
          f"por função: mediana {st.median(pct) * 100:.1f}%, p10 {q(pct,.1) * 100:.1f}%, p90 {q(pct,.9) * 100:.1f}%; com 100% certas: {sum(p == 1 for p in pct)}")
    print(f"  nós trocados: {sum(r['nos_trocados'] for r in o)}/{sum(r['nos_comuns'] for r in o)} = {sum(r['nos_trocados'] for r in o) / sum(r['nos_comuns'] for r in o) * 100:.1f}%; "
          f"de outro tipo: {sum(r['outro_tipo'] for r in o) / sum(r['nos_comuns'] for r in o) * 100:.1f}%")
    print(f"  rotação pura: {sum(r['rotacao'] for r in o)}/{len(o)}; m: mín {min(r['m'] for r in o)}, mediana {st.median(r['m'] for r in o)}, máx {max(r['m'] for r in o)}; valores distintos {len({r['m'] for r in o})}")
    print(f"  com nós isolados: {sum(r['isolados'] > 0 for r in o)}; com nós das arestas sem embedding: {sum(r['em_falta'] > 0 for r in o)}; ids não numéricos: {sum(r['lixo'] > 0 for r in o)}")
    print(f"  comprimentos de id: {Counter(r['digitos'] for r in o)}")
falhas = [r for r in linhas if r["estado"] != "ok"]
print("\nfalhas do loader:")
for r in falhas[:6]:
    print(" ", r["tipo"], r["nome"], "| nós .dot", r.get("nos_dot"), "txt", r.get("nos"), "| em falta", r.get("em_falta"), "sem label", r.get("falta_sem_label"), "|", r.get("falta_txt"), "|", r.get("erro"))
print("em falta, entre os ok:", Counter((r["tipo"], r["em_falta"] > 0) for r in linhas if r["estado"] == "ok"))
print("nos dot != txt:", Counter((r["tipo"], r["nos_dot"] != r["nos"]) for r in linhas if "nos" in r))
