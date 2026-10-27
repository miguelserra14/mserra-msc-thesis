"""02 - O texto de cada nó que vai para o GraphCodeBERT é o código desse nó?

Compara cada linha dos .txt gerados pelo generate_txt_force ORIGINAL com a etiqueta do nó no .dot.
Os nós IDENTIFIER e LITERAL têm 3 campos ("TIPO, linha", código próprio, código da expressão-mãe);
o código original fica com o último. Verifica também dois casos que podiam estragar os .txt
(código multilinha e strings com "<BR/>") e aplica a correção proposta.
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import (PIPELINE, TIPOS, cabecalho, campos_label, exigir_preparacao, ler_dot, pasta_pipeline, secao,
                   veredicto)

FUNCOES_DO_TESTE = {"copy_item", "helper", "log_msg", "compute", "get_len", "main"}


def texto_corrigido(label_html):
    """Correção: IDENTIFIER/LITERAL usam o 2.º campo (código do próprio nó); os restantes o último."""
    campos = campos_label(label_html)
    tipo = campos[0].split(",")[0]
    if tipo in ("IDENTIFIER", "LITERAL") and len(campos) >= 3:
        return campos[1]
    return campos[-1]


def ler_txt(caminho):
    linhas = Path(caminho).read_text(encoding="utf-8").splitlines()
    pares, invalidas = {}, []
    for l in linhas:
        partes = l.strip().split(" ", 1)          # exatamente como create_embeddings.py lê
        if len(partes) == 2 and partes[0].isdigit():
            pares[partes[0]] = partes[1]
        else:
            invalidas.append(l)
    return pares, invalidas


def main():
    cabecalho("02 - Texto dos nós enviado ao GraphCodeBERT")
    exigir_preparacao()
    errados = Counter()
    total_nos = Counter()
    linhas_invalidas = []
    exemplos = []
    textos_repetidos = 0

    for tipo in TIPOS:
        for dot in sorted(pasta_pipeline(tipo).glob("*.dot")):
            nome, labels, _ = ler_dot(dot)
            if nome not in FUNCOES_DO_TESTE:
                continue
            pares, invalidas = ler_txt(str(dot) + ".txt")
            linhas_invalidas += invalidas
            for nid, label in labels.items():
                campos = campos_label(label)
                tipo_no = campos[0].split(",")[0]
                total_nos[tipo] += 1
                if tipo_no in ("IDENTIFIER", "LITERAL") and len(campos) >= 3:
                    if pares.get(nid) != campos[1]:
                        errados[tipo] += 1
                        if tipo == "ast" and nome == "compute":
                            exemplos.append((tipo_no, campos[1], pares.get(nid), texto_corrigido(label)))
            if tipo == "ast":
                textos_repetidos += sum(c - 1 for c in Counter(pares.values()).values() if c > 1)

    secao("Nós IDENTIFIER/LITERAL com texto errado (pipeline original)")
    for tipo in TIPOS:
        print(f"{tipo}: {errados[tipo]} de {total_nos[tipo]} nós das funções do teste")
    print("\nExemplos (AST de compute):")
    print(f"{'tipo':11} {'código do nó':14} {'texto no .txt':22} texto com a correção")
    for t, proprio, atual, corr in exemplos[:10]:
        print(f"{t:11} {proprio:14} {atual:22} {corr}")

    secao("Casos que podiam partir os .txt (esperado: sem problema)")
    print(f"linhas de .txt que create_embeddings.py não consegue ler: {len(linhas_invalidas)}")
    copy_cfg = (pasta_pipeline("cfg") / "1-cfg.dot.txt").read_text(encoding="utf-8").splitlines()
    print("código multilinha (memcpy em 3 linhas no ficheiro C) fica numa linha:",
          next(l for l in copy_cfg if "memcpy(" in l))
    log_cfg = (pasta_pipeline("cfg") / "4-cfg.dot.txt").read_text(encoding="utf-8").splitlines()
    print('string com "<BR/>" dentro do código fica intacta:', next(l for l in log_cfg if "printf" in l))
    print("nota: o Joern trunca o texto de nós longos com '...' (ver a linha do memcpy acima; perde-se src->len)")

    secao("Outras perdas de informação no .txt")
    print("o tipo do nó (METHOD, CALL, IDENTIFIER, ...) e o número da linha são descartados")
    print(f"nós da AST com texto igual ao de outro nó do mesmo grafo (logo, embedding igual): {textos_repetidos}")

    n_err = sum(errados.values())
    veredicto(n_err > 0, "MÉDIA",
              f"{n_err} nós IDENTIFIER/LITERAL recebem o texto da expressão-mãe em vez do seu "
              f"(ex.: 'a' e 'b' passam a 'a + b'); afeta sobretudo a AST. Multilinha e '<BR/>' em strings não são problema.")


if __name__ == "__main__":
    main()
