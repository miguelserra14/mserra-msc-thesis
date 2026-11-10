# Interruptor do problema 01 (alinhamento nós ↔ arestas).
#
# Escolhe que código da Constança corre na extração (create_embeddings.py), no loader
# (load_graphcodebert_embeddings.py) e no treino (detect_vulnerabilities.py):
#
#   USAR_CODIGO_ORIGINAL = True   -> código original da Constança, sem a correção (baseline da tese)
#   USAR_CODIGO_ORIGINAL = False  -> código corrigido
#
# Basta mudar o valor e voltar a correr o script; não é preciso mais nada. Cada script imprime o modo
# em que está, e de onde o leu, na linha "Edge alignment: ...".
#
# Os grafos já construídos ficam em cache com nomes diferentes nos dois modos
# ({tipo}-graphcodebert-embeddings.pkl no original, ...-aligned.pkl no corrigido), por isso mudar o modo
# nunca reutiliza os grafos do outro.

USAR_CODIGO_ORIGINAL = False
