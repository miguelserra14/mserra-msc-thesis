"""Calcula as embeddings com o GraphCodeBERT REAL para os textos dos nós da amostra.

Mesmos parâmetros de create_embeddings.py (microsoft/graphcodebert-base, truncation a 512, mean pooling com
attention mask). Cada texto distinto é calculado uma só vez: os nós repetem muito texto, o que poupa tempo e RAM.
Grava em _gerado/impacto_tese/: textos.pkl (lista de textos) e vetores.npy (N x 768, float32).
"""
import ctypes
import pickle
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RAIZ))
from comum import GERADO

import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer

SAIDA = GERADO / "impacto_tese"
MODELO = "microsoft/graphcodebert-base"
LOTE = 64


def main():
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    torch.set_num_threads(6)
    textos = set()
    for p in sorted((SAIDA / "grafos").glob("*.pkl")):
        with open(p, "rb") as f:
            for g in pickle.load(f)["grafos"]:
                textos.update(t for _, t in g["nos"])
    # por comprimento: lotes com textos parecidos gastam menos tempo em padding
    textos = sorted(textos, key=lambda s: (len(s), s))
    print(f"{len(textos)} textos distintos", flush=True)

    tok = RobertaTokenizer.from_pretrained(MODELO)
    modelo = RobertaModel.from_pretrained(MODELO).eval()
    vetores = np.lib.format.open_memmap(SAIDA / "vetores.tmp.npy", mode="w+", dtype=np.float32, shape=(len(textos), 768))
    t0 = time.time()
    for i in range(0, len(textos), LOTE):
        ent = tok(textos[i:i + LOTE], return_tensors="pt", truncation=True, max_length=512, padding=True)
        with torch.no_grad():
            h = modelo(**ent).last_hidden_state
        m = ent["attention_mask"].unsqueeze(-1)
        vetores[i:i + LOTE] = ((h * m).sum(1) / m.sum(1)).numpy()
        if (i // LOTE) % 50 == 0:
            print(f"{i}/{len(textos)} {(time.time() - t0) / 60:.1f} min", flush=True)
    vetores.flush()
    del vetores
    (SAIDA / "vetores.tmp.npy").replace(SAIDA / "vetores.npy")
    with open(SAIDA / "textos.pkl", "wb") as f:
        pickle.dump(textos, f, protocol=4)
    print(f"FIM {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
