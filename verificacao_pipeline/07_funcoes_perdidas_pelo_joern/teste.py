"""07 - Todas as funções do dataset chegam a ter grafo?

Parte 1 (dados reais): mede que fração das funções vulneráveis e não vulneráveis fica sem .dot.
Parte 2 (código real do kernel): corre o Joern, como a pipeline (um ficheiro isolado, sem headers),
sobre dois ficheiros do Linux nos commits exatos do dataset, com e sem definir as anotações do kernel.
Os ficheiros estão em dados/linux_src/ (descarregados de github.com/torvalds/linux).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import DADOS, TIPOS, cabecalho, correr_joern, ler_funcoes_fonte, ler_mapeamento, nomes_dos_dots, secao, veredicto

CASOS = [
    # ficheiro em dados/linux_src, commit completo, funções do dataset que ficaram sem grafo
    ("ip_sockglue@534afb90.c", "534afb90a9cd0b9643f62d660c164e1d924f39cf", ["ip_setsockopt", "ip_getsockopt"]),
    ("ioremap@1f5ee8da.c", "1f5ee8da005f50d9f46ae5a7edba9a9c2d37b32e", ["iounmap", "__ioremap"]),
]
DEFINES = ["--define", "__user", "--define", "__iomem"]


def main():
    cabecalho("07 - Funções que o Joern não exporta (e que por isso desaparecem do dataset)")

    secao("Parte 1 - dados reais: funções do dataset sem grafo")
    fn = ler_funcoes_fonte()
    fn = fn.drop_duplicates(["commit", "file", "name"])
    taxas = {}
    for tipo in TIPOS:
        m = ler_mapeamento(tipo)
        com_grafo = set(zip(m[m.CFG_filepath.notna()].commit, m[m.CFG_filepath.notna()].filepath,
                            m[m.CFG_filepath.notna()].function_name))
        tem = [k in com_grafo for k in zip(fn.commit, fn.file, fn.name)]
        fn[f"tem_{tipo}"] = tem
        v = fn[fn.vuln][f"tem_{tipo}"]
        n = fn[~fn.vuln][f"tem_{tipo}"]
        taxas[tipo] = ((1 - v.mean()) * 100, (1 - n.mean()) * 100)
        print(f"{tipo}: sem grafo -> vulneráveis {int((~v).sum())}/{len(v)} ({taxas[tipo][0]:.1f}%) | "
              f"não vulneráveis {int((~n).sum())}/{len(n)} ({taxas[tipo][1]:.1f}%)")

    m = ler_mapeamento("pdg").drop_duplicates()
    m["tem"] = m.CFG_filepath.notna()
    por_ficheiro = m.groupby(["commit", "filepath"]).tem.agg(["sum", "count"])
    parcial = int(((por_ficheiro["sum"] > 0) & (por_ficheiro["sum"] < por_ficheiro["count"])).sum())
    nada = int((por_ficheiro["sum"] == 0).sum())
    print(f"\npdg, por ficheiro: {len(por_ficheiro)} ficheiros | com algumas funções em falta: {parcial} | "
          f"sem nenhuma função: {nada}")

    secao("Parte 2 - código real do kernel: Joern sem headers vs. com as anotações definidas")
    recuperadas = []
    mp = ler_mapeamento("pdg").drop_duplicates()
    for ficheiro, commit, funcoes in CASOS:
        for f in funcoes:
            linhas = mp[(mp.commit == commit) & (mp.function_name == f)]
            print(f"no CSV real: {f:16} vulnerável={linhas.vulnerable_label.iloc[0]!s:5} "
                  f"grafo={'não' if linhas.CFG_filepath.isna().all() else 'sim'}")
        caminho = DADOS / "linux_src" / ficheiro
        sem = set(nomes_dos_dots(correr_joern(caminho, ["cfg"])["cfg"]).values())
        com = set(nomes_dos_dots(correr_joern(caminho, ["cfg"], DEFINES)["cfg"]).values())
        assinatura = next(l.strip() for l in caminho.read_text(encoding="utf-8", errors="replace").splitlines()
                          if l.split("(")[0].split()[-1:] == [funcoes[0]] or f" {funcoes[0]}(" in l and "{" not in l)
        print(f"{ficheiro}:  assinatura de {funcoes[0]}: {assinatura}")
        for f in funcoes:
            print(f"   {f:16} pipeline (sem headers): {f in sem!s:5} | com {' '.join(DEFINES)}: {f in com}")
            recuperadas.append((f not in sem) and (f in com))

    v, n = taxas["pdg"]
    veredicto(all(recuperadas), "ALTA",
              f"perdem-se {v:.1f}% das funções vulneráveis contra {n:.1f}% das não vulneráveis. Causa provada em 4 "
              "funções reais: sem os headers do kernel, anotações como __user/__iomem impedem o Joern de "
              "reconhecer a função. Que fração das perdas totais vem desta causa não está medida.")


if __name__ == "__main__":
    main()
