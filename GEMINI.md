# Contexto para o Assistente de Código Gemini

Este documento fornece contexto técnico detalhado para que o assistente de IA Gemini entenda a arquitetura, o fluxo de dados e os padrões científicos deste projeto de dissertação.

---

## 🎯 Visão Geral do Projeto
Classificação automatizada da qualidade do algodão (tipagem) utilizando Deep Learning e Design Fatorial Experimental. O estudo compara arquiteturas (CNN vs. Transformers) sob diferentes condições de iluminação física e expansão espectral digital.

---

## 📂 Estrutura de Pastas e Responsabilidades

### 1. Dados (`data/`)
*   `bronze/`: Amostras brutas da algodoeira.
*   `silver/`: Imagens recortadas e normalizadas.
*   `gold/`: Datasets finais organizados em `v1` a `v12`.
    *   `datasets/{version}/train_val/`: Conjunto de treinamento e validação (80%).
    *   `datasets/{version}/test/`: Conjunto de hold-out fixo (20%) para avaliação final.

### 2. Configurações (`src/config/`)
*   `models/models/`: Arquiteturas base e hiperparâmetros padrão.
*   `models/optimized/`: **[IMPORTANTE]** YAMLs gerados pelo Optuna com as melhores configurações encontradas para cada experimento.
*   `data/versions/`: Definições das versões do dataset (filtros, canais, etc.).

### 3. Resultados e Evidências (`results/`)
*   `best_models_manifest.csv`: Guia central da elite dos modelos minerados do MLflow.
*   `weights/`: Checkpoints (.pth) dos melhores modelos organizados por versão.
*   `predictions/`: Arquivos binários (.npz) contendo y_true, y_pred e y_probs do conjunto de teste.
*   `dissertation_artifacts/`: Tabelas LaTeX, Matrizes de Confusão e Heatmaps de P-Value prontos para o `main.tex`.

---

## ⚙️ Pipeline de Pesquisa (Workflow)

### Etapa 1: Otimização (Fase Optuna)
Uso do `optimize_model.py` para busca bayesiana de hiperparâmetros. O objetivo principal é o **MCC (Matthews Correlation Coefficient)** devido ao desbalanceamento de classes.

### Etapa 2: Consolidação Científica (Fase Dissertação)
Este é o fluxo rigoroso para geração dos capítulos de resultados:
1.  **Mineração (`select_best_models.py`):** Identifica os campeões no MLflow e organiza seus YAMLs em `optimized/`.
2.  **Tribunal de Teste (`run_final_test_inference.py`):** Inferência técnica no hold-out. Mede latência (com sincronização de hardware) e complexidade (**GFLOPs** via fvcore).
3.  **Análise Estatística (`generate_hypothesis_results.py`):** Executa a **Cascata de McNemar** e gera artefatos LaTeX.

---

## 🛡️ Regras de Ouro para o Assistente
1.  **Rigor de Normalização:** Nunca avalie modelos sem carregar as estatísticas de média/std do conjunto `train_val` correspondente.
2.  **Transparência Estatística:** No heatmap de P-Values, sempre exiba o valor numérico e use o sufixo `(ns)` para $p > 0.05$.
3.  **Consistência de Canais:** Detecte automaticamente se a versão é RGB (3 canais) ou Expandida (15 canais) via nome da pasta ou YAML.
4.  **Prioridade de Métrica:** Confie no MCC e no ROC-AUC Macro acima da Acurácia Bruta.
