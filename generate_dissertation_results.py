import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    matthews_corrcoef, f1_score, cohen_kappa_score, 
    accuracy_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import friedmanchisquare
import glob
import cv2

# --- CONFIGURAÇÕES ---
PREDICTIONS_DIR = "results/predictions"
ARTIFACTS_DIR = "dissertacao/tables"
IMG_DIR = "dissertacao/images"
SUMMARY_PATH = "results/evaluation_summary.csv"
WEIGHTS_BASE_DIR = "results/weights"
DATA_BASE_DIR = "data/gold/datasets"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

CNN_MODELS = ['RESNET', 'VGG', 'INCEPTION', 'EFFICIENTNET', 'DENSENET', 'CONVNEXT']
TRANSFORMER_MODELS = ['VIT', 'SWIN']

def get_paradigm(model):
    if model.upper() in CNN_MODELS: return 'CNN'
    if model.upper() in TRANSFORMER_MODELS: return 'Transformer'
    return 'Other'

def load_all_predictions():
    files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.npz"))
    data = {}
    for f in files:
        name = os.path.basename(f).replace("_preds.npz", "")
        data[name] = np.load(f, allow_pickle=True)
    return data

def run_mcnemar(y_true, y_pred_a, y_pred_b):
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    table = np.zeros((2, 2))
    table[0, 0] = np.sum(correct_a & correct_b)
    table[0, 1] = np.sum(correct_a & ~correct_b)
    table[1, 0] = np.sum(~correct_a & correct_b)
    table[1, 1] = np.sum(~correct_a & ~correct_b)
    return mcnemar(table, exact=True).pvalue

def bold_best(df, metrics):
    df_tex = df.copy()
    for m in metrics:
        if m in df_tex.columns:
            if m in ['Latency (ms)', 'GFLOPs', 'Size (MB)']:
                best_val = df_tex[m].min()
            else:
                best_val = df_tex[m].max()
            df_tex[m] = df_tex[m].apply(lambda x: f"\\textbf{{{x:.4f}}}" if x == best_val else f"{x:.4f}")
    return df_tex

def generate_full_metrics(predictions_data, summary_df):
    results = []
    for name, d in predictions_data.items():
        y_true = d['y_true']
        y_pred = d['y_pred']
        parts = name.split('_')
        version = parts[-1].lower()
        model_type = "_".join(parts[:-1]).lower()
        
        summary_row = summary_df[(summary_df['model_type'].str.lower() == model_type) & 
                                (summary_df['dataset_version'].str.lower() == version)]
        
        latency = summary_row['latency_avg_ms'].values[0] if not summary_row.empty else 0
        gflops = summary_row['gflops'].values[0] if not summary_row.empty else 0
        params = summary_row['params_m'].values[0] if not summary_row.empty else 0
        
        weights_path = os.path.join(WEIGHTS_BASE_DIR, version, f"{model_type}_best.pth")
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024) if os.path.exists(weights_path) else 0

        results.append({
            'Model': model_type.upper(),
            'Version': version.upper(),
            'Paradigm': get_paradigm(model_type),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'F1-Macro': f1_score(y_true, y_pred, average='macro'),
            'Kappa': cohen_kappa_score(y_true, y_pred),
            'Acc': accuracy_score(y_true, y_pred),
            'Latency (ms)': latency,
            'GFLOPs': gflops,
            'Params (M)': params,
            'Size (MB)': file_size_mb
        })
    return pd.DataFrame(results)

def main():
    print("\n>>> INICIANDO GERAÇÃO DE ARTEFATOS PARA DISSERTAÇÃO")
    
    preds_data = load_all_predictions()
    summary_df = pd.read_csv(SUMMARY_PATH)
    df = generate_full_metrics(preds_data, summary_df)
    
    sns.set_theme(style="whitegrid", palette="muted")

    # --- FASE 1: BENCHMARK V1 (Tabela 4.1) ---
    print("\n[Fase 1] Gerando Benchmark V1...")
    v1_df = df[df['Version'] == 'V1'].sort_values('MCC', ascending=False)
    v1_tex = bold_best(v1_df[['Model', 'MCC', 'F1-Macro', 'Kappa', 'GFLOPs']], ['MCC', 'F1-Macro', 'Kappa', 'GFLOPs'])
    v1_tex.to_latex(os.path.join(ARTIFACTS_DIR, "metrics_table.tex"), index=False, escape=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=v1_df, x='Model', y='MCC', hue='Paradigm', dodge=False)
    plt.title("Comparação de Paradigmas (H1) - Dataset V1 (Baseline)")
    plt.savefig(os.path.join(IMG_DIR, "h1_paradigm_comparison.pdf"), bbox_inches='tight')

    # --- FASE 2: QUALITATIVA V1 (Seção 4.2) ---
    print("\n[Fase 2] Gerando Análise Qualitativa V1...")
    best_v1 = v1_df.iloc[0]['Model'].lower()
    k_best = f"{best_v1}_v1"
    
    # Matriz de Confusão
    cm = confusion_matrix(preds_data[k_best]['y_true'], preds_data[k_best]['y_pred'], normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f"Matriz de Confusão Normalizada - {best_v1.upper()} (V1)")
    plt.savefig(os.path.join(IMG_DIR, "best_model_confusion_matrix.pdf"), bbox_inches='tight')

    # McNemar Heatmap
    top_5_v1 = v1_df.head(5)['Model'].tolist()
    p_matrix = np.zeros((5, 5))
    annot_matrix = []
    for i, m1 in enumerate(top_5_v1):
        row_annots = []
        for j, m2 in enumerate(top_5_v1):
            k1, k2 = f"{m1.lower()}_v1", f"{m2.lower()}_v1"
            p_val = run_mcnemar(preds_data[k1]['y_true'], preds_data[k1]['y_pred'], preds_data[k2]['y_pred'])
            p_matrix[i,j] = p_val
            if i == j: row_annots.append("-")
            elif p_val > 0.05: row_annots.append(f"{p_val:.2f}\n(ns)")
            else: row_annots.append(f"{p_val:.2e}")
        annot_matrix.append(row_annots)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_matrix, annot=annot_matrix, fmt="", xticklabels=top_5_v1, yticklabels=top_5_v1, 
                cmap='YlGnBu_r', cbar_kws={'label': 'p-value'}, vmax=0.05)
    plt.savefig(os.path.join(IMG_DIR, "p_value_heatmap.pdf"), bbox_inches='tight')

    # --- FASE 3: LUZ (H2 - Seção 4.3) ---
    print("\n[Fase 3] Gerando Impacto da Iluminação...")
    h2_df = df[df['Version'].isin(['V1', 'V9', 'V10'])].copy()
    h2_df['Light'] = h2_df['Version'].map({'V1': 'AB', 'V9': 'AM', 'V10': 'BR'})
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=h2_df, x='Light', y='MCC', hue='Model', marker='o', sort=False)
    plt.savefig(os.path.join(IMG_DIR, "h2_lighting_impact.pdf"), bbox_inches='tight')

    # Teste de Friedman para Iluminação
    pivot_h2 = h2_df.pivot(index='Model', columns='Version', values='MCC').dropna()
    stat, p_f = friedmanchisquare(pivot_h2['V1'], pivot_h2['V9'], pivot_h2['V10'])
    ranks = pivot_h2.rank(axis=1, ascending=False).mean().sort_values()
    with open(os.path.join(ARTIFACTS_DIR, "friedman_test.tex"), "w") as f:
        f.write("\\begin{tabular}{lc}\n\\toprule\nVersão & Posto Médio \\\\\n\\midrule\n")
        for v, r in ranks.items():
            f.write(f"{v} & {r:.2f} \\\\\n")
        f.write(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Friedman $\\chi^2={stat:.2f}$ ($p={p_f:.2e}$)}} \\\\\n\\bottomrule\n\\end{tabular}")

    # --- FASE 4: ESPECTRO (H3 - Seção 4.4) ---
    print("\n[Fase 4] Gerando Impacto Espectral (V1/V8, V9/V11, V10/V12)...")
    h3_mapping = {'V1':'AB', 'V8':'AB', 'V9':'AM', 'V11':'AM', 'V10':'BR', 'V12':'BR'}
    h3_df = df[df['Version'].isin(h3_mapping.keys())].copy()
    h3_df['Light'] = h3_df['Version'].map(h3_mapping)
    h3_df['Channels'] = h3_df['Version'].apply(lambda x: 'Gabor (15ch)' if x in ['V8', 'V11', 'V12'] else 'RGB (3ch)')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=h3_df, x='Light', y='MCC', hue='Channels')
    plt.savefig(os.path.join(IMG_DIR, "h3_spectral_impact.pdf"), bbox_inches='tight')

    # --- FASE 5: VIABILIDADE (H4 - Seção 4.5) ---
    print("\n[Fase 5] Gerando Viabilidade Industrial V10...")
    v10_df = df[df['Version'] == 'V10'].sort_values('MCC', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=v10_df, x='Latency (ms)', y='MCC', hue='Paradigm', size='GFLOPs', sizes=(100, 1000))
    plt.xscale('log')
    plt.savefig(os.path.join(IMG_DIR, "efficiency_tradeoff_full.pdf"), bbox_inches='tight')

    h4_tex = bold_best(v10_df[['Model', 'MCC', 'Params (M)', 'Size (MB)', 'Latency (ms)']], ['MCC', 'Latency (ms)', 'Size (MB)'])
    h4_tex.to_latex(os.path.join(ARTIFACTS_DIR, "h4_industrial_ranking.tex"), index=False, escape=False)

    print("\n>>> ARTEFATOS GERADOS COM SUCESSO!")

if __name__ == "__main__":
    main()
