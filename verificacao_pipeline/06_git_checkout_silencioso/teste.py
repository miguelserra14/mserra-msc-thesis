"""06 - Se o `git checkout` de um commit falhar, a pipeline dá por isso?

Chama a função ORIGINAL version_management.load_commit num repositório git temporário com
(a) um commit que existe e (b) um hash que não existe, e verifica o que acontece ao HEAD,
se é lançada exceção e o que é impresso.
"""
import contextlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comum import EXTRACAO, cabecalho, secao, veredicto

sys.path.insert(0, str(EXTRACAO))
from version_management import load_commit  # noqa: E402  (código original)


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def chamar(repo, commit):
    saida = io.StringIO()
    excecao = None
    with contextlib.redirect_stdout(saida):
        try:
            load_commit(str(repo), commit)
        except Exception as e:  # noqa: BLE001
            excecao = e
    return git(repo, "rev-parse", "HEAD"), excecao, saida.getvalue()


def main():
    cabecalho("06 - Falhas do git checkout em version_management.load_commit")
    repo = Path(tempfile.mkdtemp(prefix="verif06_"))
    git(repo, "init", "-q")
    git(repo, "commit", "-q", "--allow-empty", "-m", "primeiro")
    c1 = git(repo, "rev-parse", "HEAD")
    git(repo, "commit", "-q", "--allow-empty", "-m", "segundo")
    c2 = git(repo, "rev-parse", "HEAD")
    inexistente = "0123456789abcdef0123456789abcdef01234567"

    secao("(a) checkout de um commit que existe")
    head, exc, out = chamar(repo, c1)
    aviso_no_sucesso = "[GIT][WARN]" in out
    print(f"HEAD pedido {c1[:8]} -> HEAD real {head[:8]} | exceção: {exc!r} | imprime [GIT][WARN]: {aviso_no_sucesso}")
    print("   (o git escreve 'Note: switching to ...' no stderr mesmo quando corre bem)")

    secao("(b) checkout de um hash que não existe")
    git(repo, "checkout", "-q", c2)
    head, exc, out = chamar(repo, inexistente)
    print(f"HEAD pedido {inexistente[:8]} -> HEAD real {head[:8]} (ficou no commit anterior {c2[:8]})")
    print(f"exceção: {exc!r}")
    print("linha impressa: " + next((l for l in out.splitlines() if "WARN" in l), "(nenhuma)"))
    falha_silenciosa = head == c2 and exc is None

    secao("Consequência na pipeline")
    print("main.py segue para extract_graph_per_file sem verificar o HEAD: os .dot seriam extraídos da versão")
    print("anterior do ficheiro e gravados em output-{tipo}-<commit pedido>. Como o aviso aparece também nos")
    print("checkouts bem-sucedidos, uma falha real não se distingue no log.")

    veredicto(None if falha_silenciosa else False, "MÉDIA",
              "um checkout falhado não lança erro e deixa o HEAD no commit anterior; o prefixo [GIT][WARN] também "
              "aparece em todos os checkouts bem-sucedidos. Não é possível saber se aconteceu na extração do Linux sem os logs; "
              "é uma explicação possível para o teste 05.")


if __name__ == "__main__":
    main()
