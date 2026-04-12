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
    accuracy_score, confusion_matrix, classification_report
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

    # plt.figure(figsize=(10, 6))
    # sns.barplot(data=v1_df, x='Model', y='MCC', dodge=False)
    # plt.title("Comparação entre as famílias de arquiteturas - Dataset V1 (Baseline)")
    # plt.savefig(os.path.join(IMG_DIR, "h1_paradigm_comparison.pdf"), bbox_inches='tight')

    # McNemar Heatmap
    print(v1_df)
    top_5_v1 = v1_df.head(8)['Model'].tolist()
    p_matrix = np.zeros((8, 8))
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

    # --- FASE 2: QUALITATIVA V1 (Seção 4.2) ---
    print("\n[Fase 2] Gerando Análise Qualitativa V1...")
    best_v1 = v1_df.iloc[2]['Model'].lower()
    k_best = f"{best_v1}_v1"
    print(f"Melhor modelo: {best_v1}")

    y_true = preds_data[k_best]['y_true']
    y_pred = preds_data[k_best]['y_pred']
    class_names = preds_data[k_best]['class_names']

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matriz de Confusão Normalizada - {best_v1.upper()} (V1)")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(os.path.join(IMG_DIR, "best_model_confusion_matrix.pdf"), bbox_inches='tight')

    # Relatório de classificação por classe
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
    report_df = report_df.rename(columns={'precision': 'Precisão', 'recall': 'Revocação', 'f1-score': 'Escore F1'})
    report_df = report_df[['Precisão', 'Revocação', 'Escore F1']]

    # Salva para LaTeX
    report_df.to_latex(os.path.join(ARTIFACTS_DIR, "best_model_class_metrics.tex"), index=True, float_format="%.3f")




    # --- FASE 3: LUZ (H2 - Seção 4.3) ---
    print("\n[Fase 3] Gerando Impacto da Iluminação...")
    h2_df = df[df['Version'].isin(['V1', 'V9', 'V10'])].copy()
    h2_df['Light'] = h2_df['Version'].map({'V1': 'AB', 'V9': 'AM', 'V10': 'BR'})

    # Garante a ordem categórica para a linha seguir a evolução física
    h2_df['Light'] = pd.Categorical(h2_df['Light'], categories=['AB', 'AM', 'BR'], ordered=True)

    plt.figure(figsize=(6, 5))
    sns.lineplot(data=h2_df, x='Light', y='MCC', hue='Model', marker='o', sort=True)
    plt.title("Impacto da Iluminação Física (H2) no Desempenho")
    plt.ylabel("MCC")
    plt.xlabel("Fonte de Luz")
    plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(os.path.join(IMG_DIR, "h2_lighting_impact.pdf"), bbox_inches='tight')

    # Teste de Friedman para Iluminação
    pivot_h2 = h2_df.pivot(index='Model', columns='Light', values='MCC').dropna()
    stat, p_f = friedmanchisquare(pivot_h2['AB'], pivot_h2['AM'], pivot_h2['BR'])
    ranks = pivot_h2.rank(axis=1, ascending=False).mean().sort_values()
    with open(os.path.join(ARTIFACTS_DIR, "friedman_test.tex"), "w") as f:
        f.write("\\begin{tabular}{lc}\n\\toprule\nFonte de Luz & Posto Médio \\\\\n\\midrule\n")
        for v, r in ranks.items():
            f.write(f"{v} & {r:.2f} \\\\\n")
        f.write(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Friedman $\\chi^2={stat:.2f}$ ($p={p_f:.2e}$)}} \\\\\n\\bottomrule\n\\end{{tabular}}")

    # --- FASE 4: ESPECTRO (H3 - Seção 4.4) ---
    print("\n[Fase 4] Gerando Impacto Espectral (V1/V8, V9/V11, V10/V12)...")
    h3_mapping = {'V1':'AB', 'V8':'AB', 'V9':'AM', 'V11':'AM', 'V10':'BR', 'V12':'BR'}
    h3_df = df[df['Version'].isin(h3_mapping.keys())].copy()
    h3_df['Light'] = h3_df['Version'].map(h3_mapping)
    h3_df['Channels'] = h3_df['Version'].apply(lambda x: 'Gabor (15ch)' if x in ['V8', 'V11', 'V12'] else 'RGB (3ch)')

    # Garante a ordem categórica das luzes para a linha não cruzar
    h3_df['Light'] = pd.Categorical(h3_df['Light'], categories=['AB', 'AM', 'BR'], ordered=True)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=h3_df, x='Light', y='MCC', hue='Channels', marker='o', style='Channels', markersize=10)
    plt.title("Impacto da Expansão Espectral (H3) vs. Iluminação")
    plt.ylabel("MCC Médio")
    plt.xlabel("Fonte de Iluminação")
    plt.legend(title="Configuração", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.savefig(os.path.join(IMG_DIR, "h3_spectral_impact.pdf"), bbox_inches='tight')

    # Tabela McNemar para Fase 4 (Comparando V10 vs V12 - Melhor Cenário de Luz)
    print("[Fase 4] Gerando Tabela McNemar (V10 vs V12)...")
    mcnemar_results = []
    unique_models = df['Model'].unique()
    for model in unique_models:
        k_rgb = f"{model.lower()}_v10"
        k_gabor = f"{model.lower()}_v12"
        if k_rgb in preds_data and k_gabor in preds_data:
            y_true = preds_data[k_rgb]['y_true']
            y_rgb = preds_data[k_rgb]['y_pred']
            y_gabor = preds_data[k_gabor]['y_pred']
            p_val = run_mcnemar(y_true, y_rgb, y_gabor)
            mcc_rgb = df[(df['Model'] == model) & (df['Version'] == 'V10')]['MCC'].values[0]
            mcc_gabor = df[(df['Model'] == model) & (df['Version'] == 'V12')]['MCC'].values[0]
            mcnemar_results.append({
                'Modelo': model,
                'MCC (RGB)': mcc_rgb,
                'MCC (Gabor)': mcc_gabor,
                'p-value': p_val
            })
    mcnemar_df = pd.DataFrame(mcnemar_results)
    mcnemar_df['p-value'] = mcnemar_df['p-value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    mcnemar_df.to_latex(os.path.join(ARTIFACTS_DIR, "mcnemar_gabor_vs_rgb.tex"), index=False, float_format="%.4f")

    # --- FASE 5: VIABILIDADE (H4 - Seção 4.5) ---
    print("\n[Fase 5] Gerando Viabilidade Industrial V10...")
    v10_df = df[df['Version'] == 'V10'].sort_values('MCC', ascending=False)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=v10_df, x='Latency (ms)', y='MCC', hue='Paradigm', size='GFLOPs', sizes=(100, 1000))
    plt.xscale('log')
    plt.savefig(os.path.join(IMG_DIR, "efficiency_tradeoff_full.pdf"), bbox_inches='tight')

    # Seleção e tradução da tabela industrial
    cols_subset = ['Model', 'MCC', 'Params (M)', 'Size (MB)', 'Latency (ms)', 'GFLOPs']
    metrics_to_bold = ['MCC', 'Latency (ms)', 'Size (MB)', 'GFLOPs']
    
    h4_df_subset = v10_df[cols_subset].copy()
    
    # Aplicar negrito e formatação de 4 casas decimais
    h4_tex = bold_best(h4_df_subset, metrics_to_bold)
    
    # Garantir que colunas não negritadas (como Params) também tenham 4 casas
    for col in ['Params (M)']:
        h4_tex[col] = h4_tex[col].apply(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x)

    # Traduzir cabeçalhos para Português
    h4_tex = h4_tex.rename(columns={
        'Model': 'Modelo',
        'Params (M)': 'Parâmetros (M)',
        'Size (MB)': 'Tamanho (MB)',
        'Latency (ms)': 'Latência (ms)',
        'GFLOPs': 'GFLOPs'
    })
    
    h4_tex.to_latex(os.path.join(ARTIFACTS_DIR, "h4_industrial_ranking.tex"), index=False, escape=False)

    # McNemar Heatmap para V10 (Todas as arquiteturas)
    print("[Fase 5] Gerando Heatmap McNemar para V10...")
    top_models_v10 = v10_df['Model'].tolist()
    n_models = len(top_models_v10)
    p_matrix_v10 = np.zeros((n_models, n_models))
    annot_matrix_v10 = []
    
    for i, m1 in enumerate(top_models_v10):
        row_annots = []
        for j, m2 in enumerate(top_models_v10):
            k1, k2 = f"{m1.lower()}_v10", f"{m2.lower()}_v10"
            if k1 in preds_data and k2 in preds_data:
                p_val = run_mcnemar(preds_data[k1]['y_true'], preds_data[k1]['y_pred'], preds_data[k2]['y_pred'])
                p_matrix_v10[i,j] = p_val
                if i == j: row_annots.append("-")
                elif p_val > 0.05: row_annots.append(f"{p_val:.2f}\n(ns)")
                else: row_annots.append(f"{p_val:.2e}")
            else:
                p_matrix_v10[i,j] = 1.0
                row_annots.append("N/A")
        annot_matrix_v10.append(row_annots)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_matrix_v10, annot=annot_matrix_v10, fmt="", xticklabels=top_models_v10, yticklabels=top_models_v10, 
                cmap='YlGnBu_r', cbar_kws={'label': 'p-value'}, vmax=0.05)
    plt.title("Significância Estatística (McNemar) - Dataset V10")
    plt.savefig(os.path.join(IMG_DIR, "p_value_heatmap_v10.pdf"), bbox_inches='tight')

    print("\n>>> ARTEFATOS GERADOS COM SUCESSO!")

if __name__ == "__main__":
    main()

