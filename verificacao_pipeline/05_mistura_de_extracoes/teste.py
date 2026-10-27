"""05 - O CSV de mapeamento aponta cada função para o .dot dessa função?

Nos CSV reais há .dot associados a duas funções com nomes diferentes. Como cada .dot contém uma só
função, pelo menos uma dessas linhas está errada. O teste prova primeiro, com o Joern, que isto não
pode vir de uma única extração (numeração determinística; o export recusa pastas já existentes) e
depois mede o problema nos dados reais.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import DADOS, TIPOS, cabecalho, joern_export, joern_parse, ler_mapeamento, nomes_dos_dots, secao, veredicto


def main():
    cabecalho("05 - Linhas de extrações diferentes misturadas no CSV de mapeamento")

    secao("Joern: a numeração dos .dot é determinística?")
    tmp = Path(tempfile.mkdtemp(prefix="verif05_"))
    (tmp / "vuln.c").write_bytes((DADOS / "vuln.c").read_bytes())
    joern_parse(tmp / "vuln.c", tmp / "cpg.bin")
    joern_export(tmp / "cpg.bin", "ast", tmp / "a")
    joern_export(tmp / "cpg.bin", "ast", tmp / "b")
    joern_parse(tmp / "vuln.c", tmp / "cpg2.bin")             # parse independente
    joern_export(tmp / "cpg2.bin", "ast", tmp / "c")
    na, nb, nc = nomes_dos_dots(tmp / "a"), nomes_dos_dots(tmp / "b"), nomes_dos_dots(tmp / "c")
    deterministico = na == nb == nc
    print(f"3 exportações (2 parses diferentes): mesma função em cada número? {deterministico} ({len(na)} .dot)")

    secao("Joern: o que acontece ao exportar para uma pasta que já existe?")
    rc, saida = joern_export(tmp / "cpg.bin", "ast", tmp / "a")
    linha_erro = next((l for l in saida.splitlines() if "exists" in l.lower()), saida.strip()[-200:])
    print(f"código de saída {rc}: {linha_erro.strip()}")
    recusa = rc != 0

    secao("Nos CSV reais: .dot associados a mais do que uma função")
    print(f"{'tipo':4} {'.dot com 2+ nomes':>18} {'com etiquetas diferentes':>25} {'amostras do modelo':>19}")
    partilhados_total = {}
    exemplo = None
    for tipo in TIPOS:
        m = ler_mapeamento(tipo).drop_duplicates()
        m = m[m.CFG_filepath.notna()]
        por_dot = m.groupby("CFG_filepath").agg(nomes=("function_name", "nunique"), etiquetas=("vulnerable_label", "nunique"))
        partilhados = por_dot[por_dot.nomes > 1]
        conflito = partilhados[partilhados.etiquetas > 1]
        partilhados_total[tipo] = len(partilhados)
        print(f"{tipo:4} {len(partilhados):18} {len(conflito):25} {m.CFG_filepath.nunique():19}")
        if tipo == "ast" and len(partilhados):
            caminho = partilhados.index[0]
            exemplo = m[m.CFG_filepath == caminho][["function_name", "vulnerable_label"]]
            exemplo_caminho = caminho
    if exemplo is not None:
        print(f"\nexemplo ({exemplo_caminho.split('/')[-2]}/{exemplo_caminho.split('/')[-1]}):")
        print(exemplo.to_string(index=False))

    secao("Conclusão lógica")
    print("um .dot tem um único 'digraph \"nome\"'; numa só execução o dicionário nome->.dot não pode ligar dois")
    print("nomes ao mesmo ficheiro. Com numeração determinística e sem reescrita de pastas, duas linhas com nomes")
    print("diferentes para o mesmo .dot só podem vir de extrações com entrada diferente (outro conteúdo do")
    print("ficheiro ou outra versão do Joern). Pelo menos uma das linhas de cada par está errada.")

    problema = deterministico and recusa and partilhados_total["ast"] > 0
    veredicto(problema, "MÉDIA",
              f"{partilhados_total['ast']} .dot na AST e {partilhados_total['pdg']} no PDG estão ligados a duas funções "
              f"(~1,7% das amostras; o CFG tem {partilhados_total['cfg']}). A causa da extração duplicada não é "
              "determinável sem os logs (ver teste 06 para uma hipótese).")


if __name__ == "__main__":
    main()
