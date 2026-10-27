"""09 - O mesmo código pode aparecer com etiqueta 1 e com etiqueta 0?

No dataset de origem, os negativos são as funções que a correção de um CVE NÃO tocou. Se a mesma
função vulnerável já existia em commits anteriores em que outro CVE mexeu no mesmo ficheiro, aparece
lá com etiqueta 0. O teste mede isto nas amostras que chegam ao modelo e confirma um caso com o
código real do kernel (dados/linux_src/sctp_socket@*.c, descarregados de github.com/torvalds/linux).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import DADOS, TIPOS, amostras_do_modelo, cabecalho, ler_funcoes_fonte, ler_mapeamento, secao, veredicto

CASO = {
    "funcao": "sctp_setsockopt_events",
    "vulneravel": ("sctp_socket@eb1cf0f8.c", "eb1cf0f8f7a9e5a6d573d5bd72c015686a042db0"),
    "anterior": ("sctp_socket@6a55617e.c", "6a55617ed5d1aa62b850de2cf66f5ede2eef4825"),
}


def main():
    cabecalho("09 - Código igual com etiquetas contraditórias")
    fn = ler_funcoes_fonte().drop_duplicates(["commit", "file", "name"])
    idx = fn.set_index(["commit", "file", "name"])

    secao("Amostras que chegam ao modelo")
    for tipo in TIPOS:
        m = amostras_do_modelo(tipo).join(idx[["sig", "nlines", "date"]], on=["commit", "filepath", "function_name"])
        m["gemea"] = list(zip(m.filepath, m.function_name, m.sig, m.nlines))
        neg = m[m.y == 0]
        datas_neg = neg.groupby("gemea").date.min()
        pos = m[m.y == 1]
        com_gemea_0 = pos[pos.gemea.isin(set(neg.gemea))]
        anterior = (com_gemea_0.date.values > datas_neg.reindex(com_gemea_0.gemea).values).sum()
        print(f"{tipo}: positivos {len(pos)} | com gémea quase idêntica etiquetada 0: {len(com_gemea_0)} "
              f"({len(com_gemea_0) / len(pos) * 100:.1f}%) | dos quais a gémea 0 é de um commit ANTERIOR: {anterior}")

    secao(f"Confirmação com o código real: {CASO['funcao']} (net/sctp/socket.c)")
    codigo = {}
    for chave in ("vulneravel", "anterior"):
        ficheiro, commit = CASO[chave]
        linha = fn[(fn.commit == commit) & (fn.name == CASO["funcao"])].iloc[0]
        texto = (DADOS / "linux_src" / ficheiro).read_text(encoding="utf-8", errors="replace").splitlines()
        codigo[chave] = texto[linha.l0 - 1:linha.l1]
        print(f"commit {commit[:8]} ({linha.date}) linhas {linha.l0}-{linha.l1}: etiqueta {int(linha.vuln)}")
    identico = codigo["vulneravel"] == codigo["anterior"]
    print("\n" + "\n".join(codigo["vulneravel"]))
    print(f"\ncódigo byte a byte igual nas duas versões: {identico}")

    m = ler_mapeamento("pdg").drop_duplicates()
    caso = m[m.function_name == CASO["funcao"]]
    print(f"no dataset aparece em {len(caso)} commits: etiqueta 1 em {int((caso.vulnerable_label.astype(str) == 'True').sum())}, "
          f"etiqueta 0 em {int((caso.vulnerable_label.astype(str) == 'False').sum())}; "
          f"com grafo em {int(caso.CFG_filepath.notna().sum())} (a assinatura usa __user -> ver teste 07)")

    veredicto(identico, "ALTA",
              "cerca de 27% dos positivos têm uma cópia quase idêntica etiquetada 0, na maioria de commits anteriores "
              "(o mesmo código vulnerável antes de ser corrigido). Confirmado byte a byte num caso real. É uma "
              "propriedade da forma como o dataset define negativos, não um erro de código.")


if __name__ == "__main__":
    main()
