import git
import json
import os
import pandas as pd

data_directory = "function-data"


def load_commit(repository_directory, commit):
    import subprocess
    import os
    import git

    running_directory = os.getcwd()
    os.chdir(repository_directory)

    print(f"Repository Directory: {repository_directory}")

    try:
        # Limpeza segura antes do checkout
        subprocess.run(["git", "reset", "--hard"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "clean", "-fd"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        repo = git.Repo(repository_directory)
        current_commit = repo.head.commit
        print(f"Current commit before checkout: {current_commit}\n")

        # Checkout do commit
        checkout_result = subprocess.run(["git", "checkout", commit], capture_output=True, text=True)
        print(checkout_result.stdout)
        if checkout_result.stderr:
            print("[GIT][WARN]", checkout_result.stderr.strip())

        print(f"Repository changed head to commit: {repo.head.commit}\n")

    except Exception as e:
        print(f"[ERROR] Git operation failed: {e}")

    finally:
        os.chdir(running_directory)

    return repo.head.commit


