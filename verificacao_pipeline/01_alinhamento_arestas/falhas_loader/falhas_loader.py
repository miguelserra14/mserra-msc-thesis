"""Mostra os nós sem texto nas falhas do loader (AST e PDG) em código real do kernel."""
import contextlib, io, re, sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aprendizagem"))
import teste_aprofundado as A
from comum import DADOS, correr_joern, ler_dot, campos_label
import torch

tipos_vazios = Counter()
for fonte in sorted((DADOS / "linux_src").glob("*.c")):
    pastas = correr_joern(fonte, ["ast", "pdg"])
    for tipo, pasta in pastas.items():
        for dot in pasta.glob("*.dot"):
            nome, labels, arestas = ler_dot(dot)
            if not any(re.match(r"METHOD, \d+", campos_label(l)[0]) for l in labels.values()) or nome.startswith(("<", "&lt;")):
                continue
            # nós cuja última parte da etiqueta fica vazia: a linha do .txt só tem o id e é descartada
            vazios = {n: l for n, l in labels.items() if not campos_label(l)[-1].strip()}
            for l in vazios.values():
                tipos_vazios[(tipo, campos_label(l)[0].split(",")[0])] += 1
            emb = {n: torch.zeros(4) for n in labels if n not in vazios}
            ei, n2i = A.parse_dot_edges(str(dot))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                g = A.create_dgl_graph_from_embedding(dict(embeddings=emb, edge_index=ei, node_to_idx=n2i))
            if tipo == "pdg" and g is None:
                sem_label = set(n2i) - set(labels)
                print("PDG", fonte.name, nome, "| nós", len(labels), "ids nas arestas", len(n2i), "| ids das arestas sem nó declarado", len(sem_label), "| vazios", len(vazios))
                print("   erro:", buf.getvalue().strip()[:150])
                texto = dot.read_text(encoding="utf-8", errors="replace").splitlines()
                print("   linhas do .dot que não são nem nó nem aresta:", [t[:100] for t in texto[1:] if t.strip() not in ("}", 'node [shape="rect"];') and not re.match(r'^"\d+" \[label', t.strip()) and not re.match(r'^\s*"\d+"\s*->', t)][:4])
                if sem_label:
                    print("   exemplo:", [t[:120] for t in texto if next(iter(sem_label)) in t][:3])
            if tipo == "ast" and vazios and tipos_vazios[("mostrado", "")] < 2:
                tipos_vazios[("mostrado", "")] += 1
                n = next(iter(vazios))
                print("AST", nome, "exemplo de nó vazio:", repr(vazios[n]))
print("tipos de nó com texto vazio:", tipos_vazios)
