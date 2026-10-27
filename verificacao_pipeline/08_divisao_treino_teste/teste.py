"""08 - O conjunto de teste é independente do conjunto de treino?

Reconstrói as amostras que chegam ao modelo (um .pt por caminho de .dot, primeira linha do CSV) e aplica
exatamente a divisão de detect_vulnerabilities.py: train_test_split(test_size=0.3, stratify=label,
random_state=42+11*i) para as 5 seeds. Mede quantas amostras de teste têm uma "gémea" no treino.

Gémea = mesmo ficheiro + mesma função noutro commit. Gémea quase idêntica = além disso, mesma assinatura
e mesmo número de linhas (aproximação de "mesmo código"; o teste 09 confirma um caso com o código real).
Nota: na pipeline real a ordem das amostras é a de os.listdir sobre os .pt; aqui é a do CSV. As
proporções são as mesmas em média, os valores exatos por seed não.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import CLASSIFICACAO, TIPOS, amostras_do_modelo, atribuicoes_de_topo, cabecalho, ler_funcoes_fonte, secao, veredicto

from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEEDS = [42 + i * 11 for i in range(5)]   # SEED_LIST de detect_vulnerabilities.py


def preparar(tipo, fn):
    m = amostras_do_modelo(tipo)
    sig = fn.drop_duplicates(["commit", "file", "name"]).set_index(["commit", "file", "name"])[["sig", "nlines"]]
    m = m.join(sig, on=["commit", "filepath", "function_name"])
    m["func"] = list(zip(m.filepath, m.function_name))
    m["gemea"] = list(zip(m.filepath, m.function_name, m.sig, m.nlines))
    return m


def medir(tr, te):
    tv, tn = te[te.y == 1], te[te.y == 0]
    gemeas_tr = set(tr.gemea)
    gemeas_tr_vuln = set(tr[tr.y == 1].gemea)
    return {
        "pos_mesma_funcao": tv.func.isin(set(tr.func)).mean(),
        "pos_quase_identica": tv.gemea.isin(gemeas_tr).mean(),
        "pos_quase_identica_vuln": tv.gemea.isin(gemeas_tr_vuln).mean(),
        "neg_quase_identica": tn.gemea.isin(gemeas_tr).mean(),
    }


def media(resultados):
    return {k: sum(r[k] for r in resultados) / len(resultados) for k in resultados[0]}


def main():
    cabecalho("08 - Independência entre treino e teste")
    cfg = atribuicoes_de_topo(CLASSIFICACAO / "detect_vulnerabilities.py")
    print(f"detect_vulnerabilities.py: N_RUNS={cfg.get('N_RUNS')}, divisão aleatória por amostra (train_test_split, "
          "test_size=0.3, stratify=label); o DataFrame de treino não tem commit nem ficheiro, só 'graphs' e 'label'")
    fn = ler_funcoes_fonte()

    resumo = {}
    for tipo in TIPOS:
        m = preparar(tipo, fn)
        secao(f"{tipo.upper()}: {len(m)} amostras, {m.func.nunique()} pares (ficheiro, função) distintos")
        resultados = []
        for seed in SEEDS:
            tr, te = train_test_split(m, test_size=0.3, stratify=m.y, random_state=seed)
            resultados.append(medir(tr, te))
        r = media(resultados)
        resumo[tipo] = r
        print(f"positivos de teste com a mesma função no treino:              {r['pos_mesma_funcao'] * 100:5.1f}%")
        print(f"positivos de teste com gémea quase idêntica no treino:        {r['pos_quase_identica'] * 100:5.1f}%")
        print(f"   ... e essa gémea etiquetada como vulnerável:               {r['pos_quase_identica_vuln'] * 100:5.1f}%")
        print(f"negativos de teste com gémea quase idêntica no treino:        {r['neg_quase_identica'] * 100:5.1f}%")

        # Correção: dividir por grupo (ficheiro, função) — nenhuma função aparece nos dois lados
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        i_tr, i_te = next(gss.split(m, m.y, groups=m.filepath + "::" + m.function_name))
        rc = medir(m.iloc[i_tr], m.iloc[i_te])
        print(f"com divisão por grupo (correção): positivos com gémea {rc['pos_quase_identica'] * 100:.1f}%, "
              f"negativos com gémea {rc['neg_quase_identica'] * 100:.1f}%, positivos no teste "
              f"{m.iloc[i_te].y.mean() * 100:.2f}% vs {m.y.mean() * 100:.2f}% no total")

    secao("Leitura")
    print("- negativos: mais de metade tem uma cópia quase idêntica no treino, quase sempre também negativa ->")
    print("  ficam mais fáceis de acertar (a accuracy > 95% vem sobretudo do desequilíbrio: 97,4% são negativos).")
    print("- positivos: um quarto tem uma cópia quase idêntica no treino, mas quase sempre etiquetada 0 ->")
    print("  o modelo é ensinado a dar 0 a esse código; contribui para o recall baixo (ver teste 09).")
    print("- o efeito não é só 'resultados otimistas': a avaliação não mede generalização a código novo.")

    r = resumo["pdg"]
    veredicto(True, "ALTA",
              f"no PDG, {r['neg_quase_identica'] * 100:.0f}% dos negativos e {r['pos_quase_identica'] * 100:.0f}% dos "
              "positivos de teste têm uma gémea quase idêntica no treino. Dividir por grupo (ficheiro, função) "
              "elimina-o (demonstrado); uma divisão temporal é a alternativa mais realista.")


if __name__ == "__main__":
    main()
