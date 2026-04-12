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
from matplotlib.lines import Line2D
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
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

    # --- FASE 1: BENCHMARK V1 ---
    print("\n[Fase 1] Gerando Benchmark V1...")
    v1_df = df[df['Version'] == 'V1'].sort_values('MCC', ascending=False)
    v1_tex = bold_best(v1_df[['Model', 'MCC', 'F1-Macro', 'Kappa', 'GFLOPs']], ['MCC', 'F1-Macro', 'Kappa', 'GFLOPs'])
    v1_tex.to_latex(os.path.join(ARTIFACTS_DIR, "metrics_table.tex"), index=False, escape=False)

    # --- FASE 2: QUALITATIVA V1 ---
    print("\n[Fase 2] Gerando Análise Qualitativa V1...")
    best_v1 = v1_df.iloc[2]['Model'].lower()
    k_best = f"{best_v1}_v1"
    y_true_v1 = preds_data[k_best]['y_true']
    y_pred_v1 = preds_data[k_best]['y_pred']
    classes_v1 = preds_data[k_best]['class_names']

    cm = confusion_matrix(y_true_v1, y_pred_v1, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes_v1, yticklabels=classes_v1)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.savefig(os.path.join(IMG_DIR, "best_model_confusion_matrix.pdf"), bbox_inches='tight')

    report = classification_report(y_true_v1, y_pred_v1, target_names=classes_v1, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
    report_df = report_df.rename(columns={'precision': 'Precisão', 'recall': 'Revocação', 'f1-score': 'Escore F1'})
    report_df[['Precisão', 'Revocação', 'Escore F1']].to_latex(os.path.join(ARTIFACTS_DIR, "best_model_class_metrics.tex"), index=True, float_format="%.3f")

    # --- FASE 3: LUZ (H2) ---
    print("\n[Fase 3] Gerando Impacto da Iluminação...")
    h2_df = df[df['Version'].isin(['V1', 'V9', 'V10'])].copy()
    h2_df['Light'] = h2_df['Version'].map({'V1': 'AB', 'V9': 'AM', 'V10': 'BR'})
    h2_df['Light'] = pd.Categorical(h2_df['Light'], categories=['AB', 'AM', 'BR'], ordered=True)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=h2_df, x='Light', y='MCC', hue='Model', marker='o', sort=True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(IMG_DIR, "h2_lighting_impact.pdf"), bbox_inches='tight')

    pivot_h2 = h2_df.pivot(index='Model', columns='Light', values='MCC').dropna()
    stat, p_f = friedmanchisquare(pivot_h2['AB'], pivot_h2['AM'], pivot_h2['BR'])
    ranks = pivot_h2.rank(axis=1, ascending=False).mean().sort_values()
    with open(os.path.join(ARTIFACTS_DIR, "friedman_test.tex"), "w") as f:
        f.write("\\begin{tabular}{lc}\n\\toprule\nFonte de Luz & Posto Médio \\\\\n\\midrule\n")
        for v, r in ranks.items(): f.write(f"{v} & {r:.2f} \\\\\n")
        f.write(f"\\midrule\n\\multicolumn{{2}}{{l}}{{Friedman $\\chi^2={stat:.2f}$ ($p={p_f:.2e}$)}} \\\\\n\\bottomrule\n\\end{{tabular}}")

    # --- FASE 4: ESPECTRO (H3) ---
    print("\n[Fase 4] Gerando Impacto Espectral...")
    h3_mapping = {'V1':'AB', 'V8':'AB', 'V9':'AM', 'V11':'AM', 'V10':'BR', 'V12':'BR'}
    h3_df = df[df['Version'].isin(h3_mapping.keys())].copy()
    h3_df['Light'] = h3_df['Version'].map(h3_mapping)
    h3_df['Channels'] = h3_df['Version'].apply(lambda x: 'Gabor (15ch)' if x in ['V8', 'V11', 'V12'] else 'RGB (3ch)')
    h3_df['Light'] = pd.Categorical(h3_df['Light'], categories=['AB', 'AM', 'BR'], ordered=True)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=h3_df, x='Light', y='MCC', hue='Channels', marker='o', style='Channels', markersize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(IMG_DIR, "h3_spectral_impact.pdf"), bbox_inches='tight')

    mcnemar_results = []
    for m in df['Model'].unique():
        k1, k2 = f"{m.lower()}_v10", f"{m.lower()}_v12"
        if k1 in preds_data and k2 in preds_data:
            p = run_mcnemar(preds_data[k1]['y_true'], preds_data[k1]['y_pred'], preds_data[k2]['y_pred'])
            m1 = df[(df['Model']==m) & (df['Version']=='V10')]['MCC'].values[0]
            m2 = df[(df['Model']==m) & (df['Version']=='V12')]['MCC'].values[0]
            mcnemar_results.append({'Modelo': m, 'MCC (RGB)': m1, 'MCC (Gabor)': m2, 'p-value': p})
    pd.DataFrame(mcnemar_results).to_latex(os.path.join(ARTIFACTS_DIR, "mcnemar_gabor_vs_rgb.tex"), index=False, float_format="%.4f")

    # --- FASE 5: VIABILIDADE (H4) ---
    print("\n[Fase 5] Gerando Scatter Plot Geral e Tabela Industrial...")
    
    tradeoff_df = df[df['Version'].isin(h3_mapping.keys())].copy()
    tradeoff_df['Luz'] = tradeoff_df['Version'].map(h3_mapping)
    tradeoff_df['Canais'] = tradeoff_df['Version'].apply(lambda x: '3 Canais' if x in ['V1','V9','V10'] else '15 Canais')
    
    # Reduzi a largura da figura para não espremer a plotagem
    plt.figure(figsize=(11, 8))
    # Mapeamento de cores solicitado: BR: Azul, AM: Amarelo, AB: Vermelho
    light_colors = {'BR': '#3498db', 'AM': '#f1c40f', 'AB': '#e74c3c'}
    
    models = sorted(tradeoff_df['Model'].unique())
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'h']
    model_to_marker = dict(zip(models, markers))
    
    # 3 Canais: Sólidos
    sub_3 = tradeoff_df[tradeoff_df['Canais'] == '3 Canais']
    for _, row in sub_3.iterrows():
        plt.scatter(row['Latency (ms)'], row['MCC'], 
                    s=row['GFLOPs']*40, 
                    marker=model_to_marker[row['Model']],
                    color=light_colors[row['Luz']], 
                    alpha=0.7)
    
    # 15 Canais: Vazados (apenas borda)
    sub_15 = tradeoff_df[tradeoff_df['Canais'] == '15 Canais']
    for _, row in sub_15.iterrows():
        plt.scatter(row['Latency (ms)'], row['MCC'], 
                    s=row['GFLOPs']*40, 
                    marker=model_to_marker[row['Model']],
                    facecolors='none', edgecolors=light_colors[row['Luz']], 
                    linewidths=1.5, alpha=0.8)

    # Removi plt.xscale('log') conforme solicitado
    plt.title("Trade-off Industrial: MCC vs. Latência vs. Complexidade")
    plt.xlabel("Latência de Inferência (ms)")
    plt.ylabel("MCC")

    # LEGENDA PERSONALIZADA EM BLOCOS
    legend_elements = []
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Fonte\ de\ Luz}$'))
    for l, c in light_colors.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=l, markerfacecolor=c, markersize=10))
    legend_elements.append(Line2D([0], [0], color='w', alpha=0, label='')) 
    
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Config.\ Canais}$'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='3 Canais (Sólido)', markerfacecolor='gray', markersize=10))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='15 Canais (Borda)', markerfacecolor='none', markeredgecolor='gray', markeredgewidth=1.5, markersize=10))
    legend_elements.append(Line2D([0], [0], color='w', alpha=0, label='')) 
    
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Complexidade\ (GFLOPs)}$'))
    for g in [1, 5, 15]:
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{g} GFLOPs', markerfacecolor='gray', markersize=np.sqrt(g*40)))
    legend_elements.append(Line2D([0], [0], color='w', alpha=0, label='')) 
    
    legend_elements.append(Line2D([0], [0], color='w', label=r'$\bf{Arquitetura}$'))
    for m in models:
        legend_elements.append(Line2D([0], [0], marker=model_to_marker[m], color='w', label=m, markerfacecolor='gray', markersize=10))
    
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "efficiency_tradeoff_full.pdf"), bbox_inches='tight')

    # Tabela Industrial V10
    v10_df = df[df['Version'] == 'V10'].sort_values('MCC', ascending=False)
    h4_tex = bold_best(v10_df[['Model', 'MCC', 'Params (M)', 'Size (MB)', 'Latency (ms)', 'GFLOPs']].copy(), ['MCC', 'Latency (ms)', 'Size (MB)', 'GFLOPs'])
    h4_tex['Params (M)'] = h4_tex['Params (M)'].apply(lambda x: f"{float(x):.4f}")
    h4_tex = h4_tex.rename(columns={'Model': 'Modelo', 'Params (M)': 'Parâmetros (M)', 'Size (MB)': 'Tamanho (MB)', 'Latency (ms)': 'Latência (ms)', 'GFLOPs': 'GFLOPs'})
    h4_tex.to_latex(os.path.join(ARTIFACTS_DIR, "h4_industrial_ranking.tex"), index=False, escape=False)

    # Heatmap McNemar V10
    top_v10 = v10_df['Model'].tolist()
    p_mat = np.zeros((len(top_v10), len(top_v10)))
    ann = []
    for i, m1 in enumerate(top_v10):
        r_ann = []
        for j, m2 in enumerate(top_v10):
            p = run_mcnemar(preds_data[f"{m1.lower()}_v10"]['y_true'], preds_data[f"{m1.lower()}_v10"]['y_pred'], preds_data[f"{m2.lower()}_v10"]['y_pred'])
            p_mat[i,j] = p
            r_ann.append("-" if i==j else (f"{p:.2f}\n(ns)" if p>0.05 else f"{p:.2e}"))
        ann.append(r_ann)
    plt.figure(figsize=(10, 8))
    sns.heatmap(p_mat, annot=ann, fmt="", xticklabels=top_v10, yticklabels=top_v10, cmap='YlGnBu_r', vmax=0.05)
    plt.title("Significância Estatística (McNemar) - Dataset V10")
    plt.savefig(os.path.join(IMG_DIR, "p_value_heatmap_v10.pdf"), bbox_inches='tight')

    print("\n>>> ARTEFATOS GERADOS COM SUCESSO!")

if __name__ == "__main__":
    main()
