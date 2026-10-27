# 06 — Um `git checkout` falhado passa despercebido

**Veredicto: RISCO — comportamento confirmado, ocorrência não confirmável · Gravidade: MÉDIA**

> Se o checkout de um commit falhar, `load_commit` não lança erro e o HEAD fica no commit anterior. A extração segue e grava os grafos da versão errada com o hash pedido. O aviso impresso não distingue falhas de sucessos. Não é possível saber se isto aconteceu na extração do Linux.

## Pergunta

Se o `git checkout <commit>` falhar, a pipeline pára ou avisa de forma visível?

## Como o teste funciona

Cria um repositório git temporário com dois commits e chama a função **original** [`version_management.load_commit`](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/version_management.py#L9) com:

- (a) um commit que existe;
- (b) um hash que não existe.

Regista o HEAD resultante, se há exceção e o que é impresso.

## Resultado

| Caso | HEAD depois | Exceção | Impresso |
|---|---|---|---|
| (a) commit existe | o pedido | nenhuma | `[GIT][WARN] Note: switching to '…'` |
| (b) hash não existe | **o commit anterior** | **nenhuma** | `[GIT][WARN] fatal: reference is not a tree: …` |

## Porque acontece

[version_management.py:29-32](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/version_management.py#L29-L32):

- o `subprocess.run(["git", "checkout", commit])` não usa `check=True`, e o código de saída (128 no caso (b)) nunca é lido;
- o código imprime `[GIT][WARN]` sempre que há algo no stderr, e o git escreve no stderr também nos checkouts bem-sucedidos (`Note: switching to…`). Num log com milhares de commits, as falhas ficam escondidas entre avisos normais;
- [main.py](../../uc.-msc-constanca-graphcodebert-vulnerability-detection/extraction/main.py#L55-L65) não confirma o HEAD antes de extrair.

## Impacto nos resultados da tese

Se aconteceu, alguns commits têm grafos de outra versão dos ficheiros: funções em falta, funções a mais, ou código já corrigido etiquetado como vulnerável. É uma explicação possível para a mistura de extrações do teste 05, mas não está provada. Num clone completo do repositório `torvalds/linux` todos os commits do dataset deviam existir; o risco é maior com clones parciais (`--depth`) ou com o repositório num estado inesperado.

## Como resolver

Falhar alto e confirmar o HEAD:

```python
def load_commit(repository_directory, commit):
    subprocess.run(["git", "-C", repository_directory, "reset", "--hard"], check=True, capture_output=True)
    subprocess.run(["git", "-C", repository_directory, "clean", "-fd"], check=True, capture_output=True)
    r = subprocess.run(["git", "-C", repository_directory, "checkout", "--force", commit],
                       capture_output=True, text=True)
    head = git.Repo(repository_directory).head.commit.hexsha
    if r.returncode != 0 or not head.startswith(commit):
        raise RuntimeError(f"checkout de {commit} falhou (HEAD={head}): {r.stderr.strip()}")
    return head
```

E, em `main.py`, registar num CSV de log o commit pedido, o HEAD obtido e a versão do Joern de cada extração.

## Limitações desta verificação

Confirma o comportamento do código, não que tenha havido falhas na extração real. Para isso seriam precisos os logs da Constança, ou voltar a extrair alguns commits e comparar.

## Como correr

```powershell
C:\Users\migue\miniconda3\envs\graphcodebert\python.exe 06_git_checkout_silencioso\teste.py
```

Não precisa da preparação. A saída completa está em [resultado.txt](resultado.txt).
