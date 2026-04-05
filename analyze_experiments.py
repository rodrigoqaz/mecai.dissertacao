import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

MLRUNS_DIR = "mlruns"
ANALYSIS_DIR = "results/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def parse_mlflow_run(run_path: str) -> Optional[Dict]:
    params_path = os.path.join(run_path, "params")
    metrics_path = os.path.join(run_path, "metrics")
    
    if not os.path.exists(params_path) or not os.path.exists(metrics_path):
        return None

    data = {'run_id': os.path.basename(run_path)}
    
    # Extrair Parâmetros
    for p in os.listdir(params_path):
        try:
            with open(os.path.join(params_path, p), "r") as f:
                data[p] = f.read().strip()
        except Exception: continue
            
    # Extrair Métricas Recursivamente (Captura subpastas final/, class/, val/, etc)
    def walk_metrics(path, prefix=""):
        if not os.path.exists(path): return
        for m in os.listdir(path):
            m_path = os.path.join(path, m)
            if os.path.isdir(m_path):
                walk_metrics(m_path, prefix + m + "/")
            else:
                try:
                    with open(m_path, "r") as f:
                        lines = f.readlines()
                        if lines:
                            # Pega o último valor logado (formato MLflow: timestamp valor step)
                            val = float(lines[-1].split()[1])
                            data[prefix + m] = val
                except Exception: continue
    
    walk_metrics(metrics_path)
    return data

def main():
    print("Iniciando análise profunda de experimentos com TODAS as métricas...")
    all_runs = []
    
    if not os.path.exists(MLRUNS_DIR):
        print("Erro: Pasta mlruns não encontrada.")
        return

    # 1. Varredura Completa de Runs v8
    for exp_id in os.listdir(MLRUNS_DIR):
        exp_path = os.path.join(MLRUNS_DIR, exp_id)
        if not os.path.isdir(exp_path) or exp_id in ["models", ".trash"]:
            continue
            
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path): continue
            
            run_data = parse_mlflow_run(run_path)
            if run_data and run_data.get('dataset_name') == "v8":
                all_runs.append(run_data)

    if not all_runs:
        print("Nenhum run v8 encontrado para análise profunda.")
        return

    df = pd.DataFrame(all_runs)
    
    # 2. Mapeamento de Métricas Críticas
    # Identifica as colunas reais presentes no seu MLflow
    metrics_map = {
        'Accuracy': 'val/accuracy' if 'val/accuracy' in df.columns else 'val/final_accuracy',
        'MCC': 'final/matthews_corrcoef',
        'Kappa': 'final/cohen_kappa',
        'ROC_AUC': 'final/macro_roc_auc',
        'F1_Macro': 'final/macro avg_f1-score'
    }

    # Filtra apenas o que realmente foi encontrado nos arquivos
    available_metrics = {k: v for k, v in metrics_map.items() if v in df.columns}
    print(f"Métricas detectadas nos experimentos: {list(available_metrics.keys())}")

    # 3. Cálculo do Score Ponderado de Excelência
    # Normalizamos entre 0 e 1 para criar um ranking justo
    for name, col in available_metrics.items():
        min_val, max_val = df[col].min(), df[col].max()
        df[f'{name}_norm'] = (df[col] - min_val) / (max_val - min_val + 1e-6)

    # Pesos do Score: Equilíbrio entre Acerto Bruto (Acc) e Capacidade de Discriminação (MCC/Kappa/ROC)
    weights = {'Accuracy_norm': 0.4, 'MCC_norm': 0.2, 'Kappa_norm': 0.2, 'ROC_AUC_norm': 0.2}
    actual_weights = {k: v for k, v in weights.items() if k in df.columns}
    total_w = sum(actual_weights.values())
    
    df['promising_score'] = sum(df[k] * (v/total_w) for k, v in actual_weights.items())

    # 4. Estatísticas Agregadas por Arquitetura
    agg_fields = {v: ['mean', 'max', 'std'] for v in available_metrics.values()}
    agg_fields[available_metrics['Accuracy']].append('count')
    
    stats = df.groupby('model_type').agg(agg_fields).reset_index()
    
    # 5. Ranking da Elite (Top 15 Runs)
    display_cols = ['model_type'] + list(available_metrics.values()) + ['promising_score']
    top_runs = df.sort_values(by='promising_score', ascending=False).head(15)[display_cols]

    # 6. Visualizações de Alta Fidelidade
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Accuracy (Boxplot)
    plt.subplot(2, 2, 1)
    sns.boxplot(data=df, x='model_type', y=available_metrics['Accuracy'], palette='Set2')
    plt.title('Dispersão de Acurácia (%)')
    plt.xticks(rotation=45)

    # Plot 2: MCC vs Accuracy (Scatter)
    if 'MCC' in available_metrics:
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x=available_metrics['Accuracy'], y=available_metrics['MCC'], hue='model_type', s=100, alpha=0.7)
        plt.title('Acurácia vs Estabilidade (MCC)')

    # Plot 3: ROC AUC (Boxplot)
    if 'ROC_AUC' in available_metrics:
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='model_type', y=available_metrics['ROC_AUC'], palette='Spectral')
        plt.title('Capacidade Discriminativa (ROC AUC)')
        plt.xticks(rotation=45)

    # Plot 4: Score de Promessa (Heatmap ou Bar)
    plt.subplot(2, 2, 4)
    avg_score = df.groupby('model_type')['promising_score'].mean().sort_values(ascending=False)
    avg_score.plot(kind='bar', color='teal', alpha=0.8)
    plt.title('Score Médio de Promessa (Ponderado)')
    plt.ylabel('Score (0-1)')

    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "deep_analysis_metrics.png"))

    # 7. Salvamento de Dados Consolidados
    df.to_csv(os.path.join(ANALYSIS_DIR, "deep_consolidated_v8.csv"), index=False)
    stats.to_csv(os.path.join(ANALYSIS_DIR, "stats_by_architecture.csv"), index=False)

    print("\n" + "="*90)
    print(f"RANKING DOS TOP RUNS (BASEADO EM {list(available_metrics.keys())})")
    print("="*90)
    print(top_runs.to_string(index=False))
    
    print("\n" + "="*90)
    print("ESTATÍSTICAS POR MODELO")
    print("="*90)
    print(stats.to_string())
    
    print(f"\nAnálise concluída. Artefatos salvos em: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()
