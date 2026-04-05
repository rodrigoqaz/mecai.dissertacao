import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

MLRUNS_DIR = "mlruns"
ANALYSIS_DIR = "results/dataset_comparison"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

def parse_mlflow_run(run_path: str) -> Optional[Dict]:
    params_path = os.path.join(run_path, "params")
    metrics_path = os.path.join(run_path, "metrics")
    
    if not os.path.exists(params_path) or not os.path.exists(metrics_path):
        return None

    data = {'run_id': os.path.basename(run_path)}
    
    for p in os.listdir(params_path):
        try:
            with open(os.path.join(params_path, p), "r") as f:
                data[p] = f.read().strip()
        except Exception: continue
            
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
                            val = float(lines[-1].split()[1])
                            data[prefix + m] = val
                except Exception: continue
    
    walk_metrics(metrics_path)
    return data

def main():
    print("Iniciando comparação evolutiva: Dataset v1 vs v8...")
    all_runs = []
    
    if not os.path.exists(MLRUNS_DIR):
        print("Erro: Pasta mlruns não encontrada.")
        return

    for exp_id in os.listdir(MLRUNS_DIR):
        exp_path = os.path.join(MLRUNS_DIR, exp_id)
        if not os.path.isdir(exp_path) or exp_id in ["models", ".trash"]: continue
            
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            if not os.path.isdir(run_path): continue
            
            run_data = parse_mlflow_run(run_path)
            if run_data and run_data.get('dataset_name') in ["v1", "v8"]:
                all_runs.append(run_data)

    if not all_runs:
        print("Nenhum run v1 ou v8 encontrado.")
        return

    df = pd.DataFrame(all_runs)
    
    acc_col = 'val/accuracy' if 'val/accuracy' in df.columns else 'val/final_accuracy'
    mcc_col = 'final/matthews_corrcoef' if 'final/matthews_corrcoef' in df.columns else None
    
    # Agregação: Melhor resultado por modelo e versão
    best_acc = df.groupby(['model_type', 'dataset_name'])[acc_col].max().unstack()
    
    # Cálculo de Ganho
    if 'v1' in best_acc.columns and 'v8' in best_acc.columns:
        best_acc['Ganho_Absoluto'] = best_acc['v8'] - best_acc['v1']
        best_acc['Ganho_Percentual'] = (best_acc['Ganho_Absoluto'] / (best_acc['v1'] + 1e-6)) * 100
    
    # Visualizações
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    df_plot = df.groupby(['model_type', 'dataset_name'])[acc_col].max().reset_index()
    sns.barplot(data=df_plot, x='model_type', y=acc_col, hue='dataset_name', palette='viridis')
    plt.title('Melhor Acurácia por Arquitetura: v1 vs v8')
    plt.ylabel('Acurácia (%)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.subplot(2, 1, 2)
    if 'Ganho_Percentual' in best_acc.columns:
        best_acc['Ganho_Percentual'].plot(kind='bar', color='teal', alpha=0.7)
        plt.title('Impacto do Dataset v8 (Melhoria % sobre v1)')
        plt.ylabel('Ganho Relativo (%)')
        plt.axhline(0, color='black', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "evolucao_v1_v8.png"))

    # Relatórios
    best_acc.to_csv(os.path.join(ANALYSIS_DIR, "comparativo_datasets.csv"))
    
    print("\n" + "="*80)
    print("RESUMO COMPARATIVO: DATASET V1 VS V8")
    print("="*80)
    print(best_acc.to_string())
    
    # Médias Globais
    v1_mean = df[df['dataset_name'] == 'v1'][acc_col].mean()
    v8_mean = df[df['dataset_name'] == 'v8'][acc_col].mean()
    
    print("\n" + "="*50)
    print("MÉDIAS GLOBAIS DE ACURÁCIA")
    print("="*50)
    print(f"Dataset v1: {v1_mean:.2f}%")
    print(f"Dataset v8: {v8_mean:.2f}%")
    print(f"Impacto Médio: {v8_mean - v1_mean:+.2f} p.p.")
    print("="*50)
    
    print(f"\nArquivos gerados em: {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()
