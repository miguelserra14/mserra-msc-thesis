"""14 - A arquitetura DGCNN funciona como descrito e as dimensões batem certo?

Corre as classes ORIGINAIS (DGCNNEncoder, DGCNNConv1DClassifier, DGCNNVGGAdapter, adjust_to_vgg, VGGnet)
sobre os grafos gerados pela preparação, carregados com o loader original, e mede:
  - formas dos tensores nos dois caminhos (Conv1D com SortPooling; VGG com AdaptiveMaxPool);
  - quantas colunas distintas saem do adaptador VGG;
  - que camadas GraphConv entram na representação final.
"""
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (CLASSIFICACAO, PIPELINE, atribuicoes_de_topo, cabecalho, carregar_definicoes, exigir_preparacao,
                   ler_ast, secao, veredicto)

import torch

sys.path.insert(0, str(CLASSIFICACAO))
from load_graphcodebert_embeddings import create_dgl_graph_from_embedding  # noqa: E402

DV = CLASSIFICACAO / "detect_vulnerabilities.py"


def main():
    cabecalho("14 - Arquitetura DGCNN (encoder, SortPooling/Conv1D, adaptador VGG)")
    exigir_preparacao()
    cfg = atribuicoes_de_topo(DV)
    d = carregar_definicoes(DV, {"DGCNNEncoder", "DGCNNConv1DClassifier", "DGCNNVGGAdapter", "adjust_to_vgg"},
                            {"dropout_rate": cfg["dropout_rate"]})
    grafos = [create_dgl_graph_from_embedding(torch.load(p, weights_only=False))
              for p in sorted((PIPELINE / "dataset_final_embeddings" / "pdg").glob("*.pt"))]
    hd, k = cfg["hidden_dimension"], cfg["k_sortpooling"]
    print(f"configuração: hidden_dimension={hd}, k_sortpooling={k}, k_amp={cfg['k_amp']}, "
          f"conv2dChannelParam={cfg['conv2dChannelParam']}; {len(grafos)} grafos PDG com "
          f"{[g.num_nodes() for g in grafos]} nós")
    torch.manual_seed(0)

    secao("Caminho Conv1D (T9/T11): SortPooling -> Conv1D")
    enc = d["DGCNNEncoder"](768, hd, k_sortpooling=k, pooling_type="sort_pooling", use_pooling=True).eval()
    clf = d["DGCNNConv1DClassifier"](input_dim=k * hd[-1]).eval()
    with torch.no_grad():
        z = enc(grafos)
        out = clf(z)
    print(f"encoder -> {tuple(z.shape)} (esperado (B, k*{hd[-1]}) = (B, {k * hd[-1]})) | classificador -> {tuple(out.shape)}")
    conv1d_ok = tuple(z.shape) == (len(grafos), k * hd[-1]) and tuple(out.shape) == (len(grafos), 2)

    secao("Caminho VGG (T10/T12): AdaptiveMaxPool2d -> padding 224x224 -> VGG11")
    enc_v = d["DGCNNEncoder"](768, hd, k_sortpooling=k, use_pooling=False).eval()
    adapt = d["DGCNNVGGAdapter"](embedding_dim=sum(hd), conv2d_channels=cfg["conv2dChannelParam"], k_amp=cfg["k_amp"]).eval()
    vgg = d["VGGnet"](in_channels=cfg["conv2dChannelParam"]).eval()
    with torch.no_grad():
        bg = enc_v(grafos)
        a = adapt(bg)
        img = d["adjust_to_vgg"](a)
        out_v = vgg(img)
    distintas = torch.unique(a[0, 0], dim=1).shape[1]
    print(f"features por nó à saída do encoder: {bg.ndata['h'].shape[1]} | adaptador -> {tuple(a.shape)} "
          f"| colunas distintas: {distintas} de {a.shape[3]}")
    print(f"padding -> {tuple(img.shape)} | VGG -> {tuple(out_v.shape)}")
    vgg_ok = tuple(out_v.shape) == (len(grafos), 2)

    secao("Que camadas entram na representação")
    fwd = next(n for n in ast.walk(ler_ast(DV)) if isinstance(n, ast.ClassDef) and n.name == "DGCNNEncoder")
    fwd = next(n for n in fwd.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    for n in ast.walk(fwd):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "batch_node_features.append":
            print(f"linha {n.lineno}: {ast.unparse(n)}  <- só depois de conv4")
    print("o DGCNN original (Zhang et al., 2018) concatena as saídas de todas as camadas (32*4 = 128 por nó) e usa")
    print("tanh; aqui só a última (32) com ReLU. O adaptador foi construído para 128 (sum(hidden_dimension)),")
    print("por isso o AdaptiveMaxPool2d replica cada coluna 4 vezes.")

    veredicto(distintas < a.shape[3], "BAIXA",
              f"as dimensões batem certo e os dois caminhos correm ({'OK' if conv1d_ok and vgg_ok else 'FALHA'}); mas "
              f"o encoder só usa a última camada e o adaptador VGG produz {a.shape[3]} colunas das quais só {distintas} "
              "são distintas. Ineficiência e desvio do DGCNN original, não um erro que invalide resultados.")


if __name__ == "__main__":
    main()
