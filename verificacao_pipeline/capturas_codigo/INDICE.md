# Capturas do código problemático

Geradas por `gerar_capturas.py` a partir dos ficheiros originais. O retângulo vermelho marca as linhas que causam o problema; o relatório de cada teste explica o porquê e a correção.

| Imagem | Problema | Ficheiro e linhas | O que está marcado |
|---|---|---|---|
| [01a_ordem_texto_edge_index.png](01a_ordem_texto_edge_index.png) | [01](../01_alinhamento_arestas/RELATORIO.md) | `extraction/create_embeddings.py` 122–139 (marcadas: 130–131) | edge_index numera os nós pela ordem de texto dos ids |
| [01b_ordem_numerica_features.png](01b_ordem_numerica_features.png) | [01](../01_alinhamento_arestas/RELATORIO.md) | `classification/load_graphcodebert_embeddings.py` 40–49 (marcadas: 43, 46) | as features são empilhadas por ordem numérica |
| [02_texto_dos_nos.png](02_texto_dos_nos.png) | [02](../02_texto_dos_nos/RELATORIO.md) | `extraction/graph_extraction.py` 39–50 (marcadas: 43) | fica sempre com o último campo da etiqueta |
| [03_mapeamento_so_por_nome.png](03_mapeamento_so_por_nome.png) | [03](../03_funcoes_mesmo_nome/RELATORIO.md) | `extraction/convert_dot_to_json.py` 158–176 (marcadas: 164–165, 175–176) | dicionário função → .dot só pelo nome |
| [04a_csv_em_modo_append.png](04a_csv_em_modo_append.png) | [04](../04_csv_duplicado_reexecucao/RELATORIO.md) | `extraction/convert_dot_to_json.py` 198–206 (marcadas: 204) | CSV aberto em modo 'a' |
| [04b_mapeamento_em_cada_commit.png](04b_mapeamento_em_cada_commit.png) | [04](../04_csv_duplicado_reexecucao/RELATORIO.md) | `extraction/main.py` 53–65 (marcadas: 65) | mapeamento chamado para todos os commits em cada execução |
| [05a_caminho_vem_da_linha_do_csv.png](05a_caminho_vem_da_linha_do_csv.png) | [05](../05_mistura_de_extracoes/RELATORIO.md) | `extraction/create_embeddings.py` 198–205 (marcadas: 199, 204) | o .pt é identificado pelo caminho da linha do CSV |
| [05b_primeira_linha_decide.png](05b_primeira_linha_decide.png) | [05](../05_mistura_de_extracoes/RELATORIO.md) | `extraction/create_embeddings.py` 239–243 (marcadas: 241–243) | caminhos repetidos são saltados: a primeira linha decide |
| [06_checkout_sem_verificacao.png](06_checkout_sem_verificacao.png) | [06](../06_git_checkout_silencioso/RELATORIO.md) | `extraction/version_management.py` 28–37 (marcadas: 29, 31–32) | código de saída do checkout nunca é lido |
| [07_parse_sem_headers.png](07_parse_sem_headers.png) | [07](../07_funcoes_perdidas_pelo_joern/RELATORIO.md) | `extraction/graph_extraction.py` 89–95 (marcadas: 90–91) | joern-parse de um ficheiro isolado, sem headers nem defines |
| [08a_divisao_por_amostra.png](08a_divisao_por_amostra.png) | [08](../08_divisao_treino_teste/RELATORIO.md) | `classification/detect_vulnerabilities.py` 636–648 (marcadas: 640–645) | divisão aleatória amostra a amostra |
| [08b_dataframe_sem_metadados.png](08b_dataframe_sem_metadados.png) | [08](../08_divisao_treino_teste/RELATORIO.md) | `classification/load_graphcodebert_embeddings.py` 122–128 (marcadas: 123–126) | DataFrame só com grafo e etiqueta |
| [09_etiqueta_do_dataset.png](09_etiqueta_do_dataset.png) | [09](../09_etiquetas_contraditorias/RELATORIO.md) | `extraction/convert_dot_to_json.py` 170–176 (marcadas: 173) | a etiqueta vem do dataset ('não tocada pela correção' = 0) |
| [10_optuna_no_teste.png](10_optuna_no_teste.png) | [10](../10_selecao_no_conjunto_de_teste/RELATORIO.md) | `classification/new_classifiers_svm_xgboost.py` 228–254 (marcadas: 246, 249–250) | o Optuna avalia no conjunto de teste |
| [11a_seed_so_na_divisao.png](11a_seed_so_na_divisao.png) | [11](../11_reprodutibilidade_do_treino/RELATORIO.md) | `classification/detect_vulnerabilities.py` 634–645 (marcadas: 644) | a seed só é usada no train_test_split |
| [11b_dataloader_sem_shuffle.png](11b_dataloader_sem_shuffle.png) | [11](../11_reprodutibilidade_do_treino/RELATORIO.md) | `classification/detect_vulnerabilities.py` 764–775 (marcadas: 770–771) | DataLoader ativo sem shuffle nem sampler |
| [12a_embeddings_de_treino.png](12a_embeddings_de_treino.png) | [12](../12_svm_sobre_embeddings_de_treino/RELATORIO.md) | `classification/detect_vulnerabilities.py` 1035–1052 (marcadas: 1038–1044) | embeddings de treino geradas pelo encoder treinado nelas |
| [12b_svm_carrega_treino.png](12b_svm_carrega_treino.png) | [12](../12_svm_sobre_embeddings_de_treino/RELATORIO.md) | `classification/new_classifiers_svm_xgboost.py` 53–59 (marcadas: 54, 58) | o SVM treina nessas embeddings e testa nas de teste |
| [13a_tipo_de_aresta_descartado.png](13a_tipo_de_aresta_descartado.png) | [13](../13_representacao_do_grafo/RELATORIO.md) | `extraction/create_embeddings.py` 113–125 (marcadas: 123–124) | só os pares de ids; o tipo de aresta é descartado |
| [13b_tokenizer_por_no.png](13b_tokenizer_por_no.png) | [13](../13_representacao_do_grafo/RELATORIO.md) | `extraction/create_embeddings.py` 81–91 (marcadas: 82–83, 91) | cada nó é tokenizado sozinho, sem data-flow |
| [14a_so_ultima_camada.png](14a_so_ultima_camada.png) | [14](../14_arquitetura_dgcnn/RELATORIO.md) | `classification/detect_vulnerabilities.py` 172–184 (marcadas: 181–182) | só a saída de conv4 é guardada |
| [14b_adaptador_vgg_pooling.png](14b_adaptador_vgg_pooling.png) | [14](../14_arquitetura_dgcnn/RELATORIO.md) | `classification/detect_vulnerabilities.py` 216–225 (marcadas: 224) | AdaptiveMaxPool2d para embedding_dim colunas |
| [14c_adaptador_vgg_128.png](14c_adaptador_vgg_128.png) | [14](../14_arquitetura_dgcnn/RELATORIO.md) | `classification/detect_vulnerabilities.py` 794–799 (marcadas: 796–797) | embedding_dim = 128, mas só chegam 32 |
