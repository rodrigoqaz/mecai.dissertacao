# Plano de Análise Experimental e Teste de Hipóteses (Refinado)

Este documento detalha o fluxo de trabalho para a consolidação dos resultados da dissertação, focado na comparação de arquiteturas (CNN vs. Transformers), espaços de iluminação (AB, AM, BR) e profundidade espectral (3 canais vs. 15 canais).

---

## 🎯 Objetivo
Validar as hipóteses de pesquisa através de um fluxo rigoroso: **Seleção da Elite (via MCC) -> Avaliação Final (Teste Hold-out) -> Cascata de Testes Estatísticos (McNemar).**

---

## 🚀 Fase 1: Identificação da Elite e Recuperação de Modelos
**Objetivo:** Encontrar o melhor representante (maior MCC) de cada arquitetura para cada versão do dataset e garantir a disponibilidade dos pesos (.pth).

### 1.1 Estratégia de Mineração (MLflow Querying)
1.  **Conexão e Filtros:**
    *   Utilizar `mlflow.search_runs` para varrer todos os experimentos.
    *   **Filtro de busca:** `params.dataset_name = '{v1, v8, v9, v10}' AND status = 'FINISHED'`.
2.  **Ranking por Versão:**
    *   Para cada tupla `(model_type, dataset_version)`, ordenar os resultados por `metrics."final/matthews_corrcoef"` de forma decrescente.
    *   Selecionar o **Top 1** de cada categoria.
3.  **Coleta de Metadados:**
    *   Extrair: Run_ID, Parâmetros Otimizados (Optuna), Acurácia de Validação e MCC.

### 1.2 Protocolo de Verificação de Artefatos
1.  **Check de Integridade:** O script verificará se o arquivo `model.pth` está presente em:
    *   `artifacts/best_model/data/model.pth` (padrão do log_model)
    *   `artifacts/final_artifacts/` (padrão de artefatos manuais)
2.  **Download Local:** Os pesos validados serão baixados para uma estrutura local organizada: `results/weights/{version}/{model_type}_best.pth`.

### 1.3 Fluxo de Recuperação (Auto-Retraining)
Caso o melhor modelo de uma categoria não possua o arquivo `.pth` salvo:
1.  **Extração de Receita:** O script extrairá o dicionário completo de parâmetros e o `augmentation_config` logados no run original do MLflow.
2.  **Geração de YAML de "Melhor Configuração":**
    *   Criar um arquivo YAML em `results/configs/recovered_{model}_{version}.yaml`.
    *   Este arquivo consolidará todas as decisões do Optuna (learning rate, batch size, augmentations) em um formato legível.
3.  **Treinamento Padronizado:**
    *   Utilizar a infraestrutura existente: carregar este YAML via `Config` e passá-lo para `train_model_v2.train_single_model()`.
    *   Isso garante que o treinamento de recuperação siga rigorosamente o mesmo pipeline dos outros modelos.
4.  **Update do Manifesto:** O novo Run_ID e o novo caminho do peso serão registrados como a "Elite Oficial", garantindo rastreabilidade total.

### 1.4 Geração do Manifesto da Elite
*   **Arquivo:** `results/best_models_manifest.csv`
*   **Campos Obrigatórios:** `model_type`, `dataset_version`, `run_id`, `mcc_val`, `weights_path`, `is_recovered_run` (bool), `config_path`.
*   **Uso Futuro:** Este manifesto servirá como a única fonte de verdade para as Fases 2 e 3.

---

## ⚖️ Fase 2: Tribunal de Teste (Execução Hold-out)
**Objetivo:** Gerar as predições definitivas sobre o conjunto de teste de 20% (dados nunca vistos) e medir a eficiência computacional.

### 2.1 Protocolo de Carregamento e Dados
1.  **Utilização do Dataset Persistido:**
    *   **Diretório de Teste:** O motor de inferência deve carregar os dados diretamente da pasta `data/gold/datasets/{version}/test/`. Esta pasta já contém o split de hold-out fixo para cada versão do experimento.
    *   **Carregamento via NPYFolderDataset:** Utilizar a classe `NPYFolderDataset` apontando especificamente para o subdiretório `test`.
2.  **Mapeamento de Versões:**
    *   O script deve associar o modelo à sua respectiva pasta de teste (ex: Modelos V8 -> `data/gold/datasets/v8/test/`).
3.  **Transformações de Avaliação:**
    *   **Consistência:** Aplicar as mesmas transformações de base definidas em `src/data/data_loader.py` (Resize 224, ToDtype, Normalize).
    *   **Desativação de Augmentation:** Garantir que nenhuma técnica de aumento de dados (CutMix, MixUp, etc.) seja aplicada durante esta fase.

### 2.2 Motor de Inferência (Technical Pipeline)
1.  **Configuração do Ambiente:**
    *   Modo de Avaliação: `model.eval()`
    *   Bloqueio de Gradientes: `with torch.no_grad():`
2.  **Warm-up:** Executar 10 iterações de inferência com dados aleatórios para estabilizar as medições de hardware.
3.  **Coleta de Dados Brutos:**
    *   Capturar os logits e converter em probabilidades via `Softmax`.
    *   Coletar `y_true` (ground truth da pasta test) e `y_pred` (argmax das probabilidades).

### 2.3 Medição de Eficiência (Benchmarking)
1.  **Latência (Inference Time):**
    *   Medir o tempo de processamento por imagem individualmente.
    *   Em GPU: Inserir `torch.cuda.synchronize()` para evitar medições parciais devido à execução assíncrona do CUDA.
2.  **Hardware Profile:**
    *   **Parâmetros:** Reportar total de parâmetros via `torchinfo`.
    *   **FLOPs:** Calcular a complexidade teórica para o input correspondente à versão do modelo (3 canais ou 15 canais).

### 2.4 Persistência de Resultados
*   **Predictions (`.npz`):** Salvar em `results/predictions/{model}_{version}_preds.npz` para posterior análise estatística na Fase 3.
*   **Relatório de Métricas:** Gerar um arquivo consolidado com Acurácia, MCC e Métricas de Eficiência para todos os modelos avaliados.

---

## 🔬 Fase 3: Cascata de Testes Estatísticos (Significância)
**Objetivo:** Aplicar o Teste de McNemar em níveis crescentes de granularidade para isolar as variáveis do estudo e gerar evidências científicas robustas.

### 3.1 Motor de Teste Estatístico (Protocolo de McNemar)
1.  **Ferramenta:** Utilizar `statsmodels.stats.contingency_tables.mcnemar` com correção de continuidade de Edwards para amostras onde a tabela de contingência apresente valores baixos.
2.  **Construção da Tabela de Contingência (2x2):**
    *   **Acerto/Acerto ($a$):** Ambos os modelos acertaram a amostra.
    *   **Acerto/Erro ($b$):** Modelo A acertou, Modelo B errou.
    *   **Erro/Acerto ($c$):** Modelo A errou, Modelo B acertou.
    *   **Erro/Erro ($d$):** Ambos os modelos erraram a amostra.
3.  **Cálculo do P-Value:** A hipótese nula ($H_0$) de que os dois classificadores têm taxas de erro iguais é rejeitada se $p < 0.05$.

### 3.2 Implementação da Cascata de Testes
1.  **Nível 1 (Impacto da Física/Canais):**
    *   Comparar pares: `(Model_X_V1, Model_X_V8)`, `(Model_X_V1, Model_X_V9)`, etc.
    *   **Meta:** Isolar se a variação de performance vem do sensor (15 canais) ou da iluminação (AM/BR).
2.  **Nível 2 (Duelo Arquitetural):**
    *   Comparar pares: `(CNN_Melhor_V8, Transformer_Melhor_V8)`.
    *   **Meta:** Validar se o viés indutivo local das CNNs é superado pela atenção global dos Transformers no domínio do algodão.
3.  **Nível 3 (Ranking Global de Significância):**
    *   Gerar uma matriz de p-values comparando o "Campeão Absoluto" contra todos os outros modelos da elite.

### 3.3 Visualização e Artefatos (Latex & Plots)
1.  **Tabela Consolidada (Pandas to LaTeX):**
    *   Gerar automaticamente o código LaTeX para uma tabela `tabular` com: Modelo, Versão, MCC, F1-Macro, Kappa, Acc, Latência e Params.
2.  **Heatmap de Matrizes de Confusão:**
    *   Plotar matrizes normalizadas pela linha (Recall) para destacar a confusão entre classes adjacentes (ex: 31.1 vs 31.2).
    *   Salvar em PDF/PNG de alta resolução (300 DPI).
3.  **Gráfico de Eficiência Industrial (Pareto-style):**
    *   Eixo X: Latência (Log scale).
    *   Eixo Y: MCC.
    *   Tamanho da bolha: Nº de Parâmetros.
    *   Cores: Agrupadas por Família de Arquitetura.
4.  **Matriz de P-Values:**
    *   Gerar um heatmap onde cada célula $(i, j)$ representa o p-value da comparação entre o modelo $i$ e o modelo $j$, com anotações de estrelas para significância ($* p<0.05, ** p<0.01$).

---

## 📊 Artefatos de Saída para a Dissertação
*   **Local:** `results/dissertation_artifacts/`
*   `table_metrics_final.tex`: Tabela principal de resultados.
*   `matrix_pvalues.pdf`: Visualização da significância estatística.
*   `efficiency_tradeoff.pdf`: Gráfico de dispersão para análise de viabilidade industrial.
*   `confusion_matrices_elite/`: Pasta com as matrizes dos 5 melhores modelos.

