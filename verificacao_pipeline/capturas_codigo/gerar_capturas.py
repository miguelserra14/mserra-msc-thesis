"""Gera imagens PNG dos troços de código da Constança que causam cada problema (para slides).

Cada imagem imita um screenshot do editor (tema escuro, números de linha) com um retângulo vermelho
simples à volta das linhas em causa, como se tivesse sido desenhado no Paint. Não tem texto extra.
Lê os ficheiros originais do repositório (não os altera); o realce de sintaxe usa o `tokenize`
da biblioteca padrão e o desenho usa o Pillow.

Uso:  C:\\Users\\migue\\miniconda3\\envs\\graphcodebert\\python.exe capturas_codigo\\gerar_capturas.py
"""
import io
import keyword
import sys
import tokenize
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import REPO  # noqa: E402

SAIDA = Path(__file__).resolve().parent
FONTES = Path("C:/Windows/Fonts")
ESCALA = 2                     # como um screenshot num ecrã de alta densidade; fica nítido num slide
TAM_FONTE = 14
MAX_COLUNAS = 140              # o resto da linha fica "fora da janela", como num screenshot real

# (ficheiro, problema, ficheiro do repo, (primeira, última linha), linhas a enquadrar, descrição p/ índice)
CAPTURAS = [
    ("01a_ordem_texto_edge_index", "01", "extraction/create_embeddings.py", (122, 139), {130, 131},
     "edge_index numera os nós pela ordem de texto dos ids"),
    ("01b_ordem_numerica_features", "01", "classification/load_graphcodebert_embeddings.py", (40, 49), {43, 46},
     "as features são empilhadas por ordem numérica"),
    ("02_texto_dos_nos", "02", "extraction/graph_extraction.py", (39, 50), {43},
     "fica sempre com o último campo da etiqueta"),
    ("03_mapeamento_so_por_nome", "03", "extraction/convert_dot_to_json.py", (158, 176), {164, 165, 175, 176},
     "dicionário função → .dot só pelo nome"),
    ("04a_csv_em_modo_append", "04", "extraction/convert_dot_to_json.py", (198, 206), {204},
     "CSV aberto em modo 'a'"),
    ("04b_mapeamento_em_cada_commit", "04", "extraction/main.py", (53, 65), {65},
     "mapeamento chamado para todos os commits em cada execução"),
    ("05a_caminho_vem_da_linha_do_csv", "05", "extraction/create_embeddings.py", (198, 205), {199, 204},
     "o .pt é identificado pelo caminho da linha do CSV"),
    ("05b_primeira_linha_decide", "05", "extraction/create_embeddings.py", (239, 243), {241, 242, 243},
     "caminhos repetidos são saltados: a primeira linha decide"),
    ("06_checkout_sem_verificacao", "06", "extraction/version_management.py", (28, 37), {29, 31, 32},
     "código de saída do checkout nunca é lido"),
    ("07_parse_sem_headers", "07", "extraction/graph_extraction.py", (89, 95), {90, 91},
     "joern-parse de um ficheiro isolado, sem headers nem defines"),
    ("08a_divisao_por_amostra", "08", "classification/detect_vulnerabilities.py", (636, 648), {640, 641, 642, 643, 644, 645},
     "divisão aleatória amostra a amostra"),
    ("08b_dataframe_sem_metadados", "08", "classification/load_graphcodebert_embeddings.py", (122, 128), {123, 124, 125, 126},
     "DataFrame só com grafo e etiqueta"),
    ("09_etiqueta_do_dataset", "09", "extraction/convert_dot_to_json.py", (170, 176), {173},
     "a etiqueta vem do dataset ('não tocada pela correção' = 0)"),
    ("10_optuna_no_teste", "10", "classification/new_classifiers_svm_xgboost.py", (228, 254), {246, 249, 250},
     "o Optuna avalia no conjunto de teste"),
    ("11a_seed_so_na_divisao", "11", "classification/detect_vulnerabilities.py", (634, 645), {644},
     "a seed só é usada no train_test_split"),
    ("11b_dataloader_sem_shuffle", "11", "classification/detect_vulnerabilities.py", (764, 775), {770, 771},
     "DataLoader ativo sem shuffle nem sampler"),
    ("12a_embeddings_de_treino", "12", "classification/detect_vulnerabilities.py", (1035, 1052), set(range(1038, 1045)),
     "embeddings de treino geradas pelo encoder treinado nelas"),
    ("12b_svm_carrega_treino", "12", "classification/new_classifiers_svm_xgboost.py", (53, 59), {54, 58},
     "o SVM treina nessas embeddings e testa nas de teste"),
    ("13a_tipo_de_aresta_descartado", "13", "extraction/create_embeddings.py", (113, 125), {123, 124},
     "só os pares de ids; o tipo de aresta é descartado"),
    ("13b_tokenizer_por_no", "13", "extraction/create_embeddings.py", (81, 91), {82, 83, 91},
     "cada nó é tokenizado sozinho, sem data-flow"),
    ("14a_so_ultima_camada", "14", "classification/detect_vulnerabilities.py", (172, 184), {181, 182},
     "só a saída de conv4 é guardada"),
    ("14b_adaptador_vgg_pooling", "14", "classification/detect_vulnerabilities.py", (216, 225), {224},
     "AdaptiveMaxPool2d para embedding_dim colunas"),
    ("14c_adaptador_vgg_128", "14", "classification/detect_vulnerabilities.py", (794, 799), {796, 797},
     "embedding_dim = 128, mas só chegam 32"),
]

# Tema escuro tipo VS Code Dark+
COR = {
    "fundo": "#1e1e1e", "gutter": "#6e7681", "texto": "#d4d4d4",
    "keyword": "#569cd6", "controlo": "#c586c0", "string": "#ce9178", "comentario": "#6a9955",
    "numero": "#b5cea8", "funcao": "#dcdcaa", "classe": "#4ec9b0", "guia": "#404040",
}
VERMELHO = (255, 0, 0)          # o vermelho por omissão do Paint
CONTROLO = {"if", "elif", "else", "for", "while", "return", "try", "except", "finally", "with", "break",
            "continue", "raise", "import", "from", "as", "yield", "pass"}
CONSTANTES = {"True", "False", "None", "self"}


def colorir_ficheiro(caminho):
    """Devolve {número_de_linha: (texto, [(pedaço, cor), ...])} para o ficheiro inteiro."""
    fonte_txt = caminho.read_text(encoding="utf-8").expandtabs(4)
    linhas = fonte_txt.splitlines()
    cores = {i + 1: [COR["texto"]] * len(l) for i, l in enumerate(linhas)}
    tokens = list(tokenize.generate_tokens(io.StringIO(fonte_txt).readline))
    tipos_string = {tokenize.STRING} | {getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
                                        if hasattr(tokenize, n)}
    for i, tok in enumerate(tokens):
        cor = None
        if tok.type == tokenize.COMMENT:
            cor = COR["comentario"]
        elif tok.type in tipos_string:
            cor = COR["string"]
        elif tok.type == tokenize.NUMBER:
            cor = COR["numero"]
        elif tok.type == tokenize.NAME:
            seguinte = tokens[i + 1] if i + 1 < len(tokens) else None
            anterior = tokens[i - 1] if i > 0 else None
            if tok.string in CONTROLO:
                cor = COR["controlo"]
            elif keyword.iskeyword(tok.string) or tok.string in CONSTANTES:
                cor = COR["keyword"]
            elif anterior is not None and anterior.string == "class":
                cor = COR["classe"]
            elif seguinte is not None and seguinte.string == "(":
                cor = COR["classe"] if tok.string[:1].isupper() else COR["funcao"]
        if cor is None:
            continue
        (l0, c0), (l1, c1) = tok.start, tok.end
        for ln in range(l0, l1 + 1):
            if ln in cores:
                ini, fim = (c0 if ln == l0 else 0), (c1 if ln == l1 else len(cores[ln]))
                for c in range(ini, min(fim, len(cores[ln]))):
                    cores[ln][c] = cor
    resultado = {}
    for n, l in enumerate(linhas, start=1):
        spans, atual, cor_atual = [], "", None
        for ch, cor in zip(l, cores[n]):
            if cor != cor_atual and atual:
                spans.append((atual, cor_atual))
                atual = ""
            atual, cor_atual = atual + ch, cor
        if atual:
            spans.append((atual, cor_atual))
        resultado[n] = (l, spans)
    return resultado


def grupos_contiguos(linhas):
    grupos, atual = [], []
    for n in sorted(linhas):
        if atual and n != atual[-1] + 1:
            grupos.append(atual)
            atual = []
        atual.append(n)
    return grupos + ([atual] if atual else [])


def desenhar(nome, rel, intervalo, enquadrar):
    linhas = colorir_ficheiro(REPO / rel)
    mono = ImageFont.truetype(str(FONTES / "consola.ttf"), TAM_FONTE * ESCALA)
    cw = mono.getlength("M")
    alt_linha = int(TAM_FONTE * ESCALA * 1.4)
    a, b = intervalo
    visiveis = list(range(a, b + 1))
    digitos = max(len(str(b)), 3)
    margem_esq = int(cw * 1.5)
    gutter = int(cw * (digitos + 2.5))
    colunas = min(max(len(linhas[n][0].rstrip()) for n in visiveis), MAX_COLUNAS) + 3
    largura = int(margem_esq + gutter + cw * colunas)
    margem_topo = int(alt_linha * 0.4)
    altura = margem_topo * 2 + alt_linha * len(visiveis)

    img = Image.new("RGB", (largura, altura), COR["fundo"])
    d = ImageDraw.Draw(img)
    x0 = margem_esq + gutter
    for i, n in enumerate(visiveis):
        y = margem_topo + i * alt_linha
        topo = y + (alt_linha - TAM_FONTE * ESCALA) // 2 - ESCALA
        d.text((margem_esq, topo), str(n).rjust(digitos), font=mono, fill=COR["gutter"])
        texto, spans = linhas[n]
        # guias de indentação, como no editor
        recuo = len(texto) - len(texto.lstrip()) if texto.strip() else 0
        for nivel in range(4, recuo, 4):
            gx = x0 + cw * nivel
            d.line([gx, y, gx, y + alt_linha], fill=COR["guia"], width=1)
        x, col = x0, 0
        for parte, cor in spans:
            for ch in parte:
                if col >= MAX_COLUNAS:
                    break
                if ch != " ":
                    d.text((x, topo), ch, font=mono, fill=cor)
                x += cw
                col += 1

    # Retângulos vermelhos, um por cada bloco de linhas seguidas, à volta do código (como no Paint)
    esp = 3 * ESCALA // 2 + 1
    for grupo in grupos_contiguos(enquadrar & set(visiveis)):
        textos = [linhas[n][0] for n in grupo]
        ini = min(len(t) - len(t.lstrip()) for t in textos)
        fim = min(max(len(t.rstrip()) for t in textos), MAX_COLUNAS)
        rx0 = int(x0 + cw * ini - cw * 0.6)
        rx1 = int(min(x0 + cw * fim + cw * 0.6, largura - esp))
        ry0 = margem_topo + (grupo[0] - a) * alt_linha - ESCALA
        ry1 = margem_topo + (grupo[-1] - a + 1) * alt_linha + ESCALA
        d.rectangle([rx0, ry0, rx1, ry1], outline=VERMELHO, width=esp)

    destino = SAIDA / f"{nome}.png"
    img.save(destino, optimize=True)
    return destino, img.size


def main():
    for antigo in SAIDA.glob("*.png"):
        antigo.unlink()
    indice = ["# Capturas do código problemático", "",
              "Geradas por `gerar_capturas.py` a partir dos ficheiros originais. O retângulo vermelho marca as linhas "
              "que causam o problema; o relatório de cada teste explica o porquê e a correção.", "",
              "| Imagem | Problema | Ficheiro e linhas | O que está marcado |", "|---|---|---|---|"]
    for nome, teste, rel, intervalo, enquadrar, descricao in CAPTURAS:
        destino, (w, h) = desenhar(nome, rel, intervalo, enquadrar)
        marcadas = ", ".join(f"{g[0]}" if len(g) == 1 else f"{g[0]}–{g[-1]}" for g in grupos_contiguos(enquadrar))
        pasta = next(p.name for p in SAIDA.parent.iterdir() if p.is_dir() and p.name.startswith(teste + "_"))
        indice.append(f"| [{destino.name}]({destino.name}) | [{teste}](../{pasta}/RELATORIO.md) "
                      f"| `{rel}` {intervalo[0]}–{intervalo[1]} (marcadas: {marcadas}) | {descricao} |")
        print(f"{destino.name:40} {w}x{h}")
    (SAIDA / "INDICE.md").write_text("\n".join(indice) + "\n", encoding="utf-8")
    print(f"\n{len(CAPTURAS)} imagens + INDICE.md em {SAIDA}")


if __name__ == "__main__":
    main()
