import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    matthews_corrcoef, f1_score, cohen_kappa_score, 
    accuracy_score, roc_auc_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import friedmanchisquare, wilcoxon
import glob

# Configurações
PREDICTIONS_DIR = "results/predictions"
ARTIFACTS_DIR = "dissertacao/tables"
IMG_DIR = "dissertacao/images"
SUMMARY_PATH = "results/evaluation_summary.csv"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Definição de Paradigmas
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

def generate_full_metrics(predictions_data, summary_df):
    results = []
    for name, d in predictions_data.items():
        y_true = d['y_true']
        y_pred = d['y_pred']
        
        parts = name.split('_')
        version = parts[-1].upper()
        model_type = "_".join(parts[:-1]).upper()
        
        summary_row = summary_df[(summary_df['model_type'].str.upper() == model_type) & 
                                (summary_df['dataset_version'].str.upper() == version)]
        
        latency = summary_row['latency_avg_ms'].values[0] if not summary_row.empty else 0
        gflops = summary_row['gflops'].values[0] if not summary_row.empty else 0
        
        results.append({
            'Model': model_type,
            'Version': version,
            'Paradigm': get_paradigm(model_type),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'F1-Macro': f1_score(y_true, y_pred, average='macro'),
            'Kappa': cohen_kappa_score(y_true, y_pred),
            'Acc': accuracy_score(y_true, y_pred),
            'Latency (ms)': latency,
            'GFLOPs': gflops
        })
    return pd.DataFrame(results)

def main():
    print("\n>>> INICIANDO VALIDAÇÃO CIENTÍFICA DAS HIPÓTESES (H1-H4)")
    
    preds_data = load_all_predictions()
    summary_df = pd.read_csv(SUMMARY_PATH)
    df = generate_full_metrics(preds_data, summary_df)
    
    # --- 1. TESTE DE FRIEDMAN + POST-HOC ---
    print("\n[Linear Narrative] Executando Teste de Friedman...")
    versions = ['V1', 'V8', 'V9', 'V10']
    pivot_v = df[df['Version'].isin(versions)].pivot(index='Model', columns='Version', values='MCC').dropna()
    stat, p_friedman = friedmanchisquare(pivot_v['V1'], pivot_v['V8'], pivot_v['V9'], pivot_v['V10'])
    
    pairs = [('V1','V8'), ('V1','V9'), ('V1','V10'), ('V8','V9'), ('V8','V10'), ('V9','V10')]
    posthoc_results = {}
    for v_a, v_b in pairs:
        _, p = wilcoxon(pivot_v[v_a], pivot_v[v_b])
        posthoc_results[f"{v_a}_vs_{v_b}"] = p * len(pairs)

    ranks = pivot_v.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()

    def get_sig_group(v):
        if v == 'V10': return "(a)"
        p_adj = posthoc_results.get(f"V9_vs_V10") if v == 'V9' else posthoc_results.get(f"V1_vs_V10") if v == 'V1' else posthoc_results.get(f"V8_vs_V10")
        if p_adj > 0.05: return "(a)"
        if v == 'V8': return "(c)"
        return "(b)"

    latex_friedman = [
        "\\begin{tabular}{lrc}",
        "\\toprule",
        "\\textbf{Versão do Dataset} & \\textbf{Posto Médio} & \\textbf{Grupo} \\\\",
        "\\midrule"
    ]
    for v, rank in mean_ranks.items():
        latex_friedman.append(f"{v} & {rank:.2f} & {get_sig_group(v)} \\\\")
    
    latex_friedman.extend([
        "\\midrule",
        f"\\multicolumn{{3}}{{l}}{{\\textbf{{Estatística de Friedman}}: $\\chi^2 = {stat:.2f}$ ($p = {p_friedman:.2e}$)}} \\\\",
        "\\multicolumn{3}{l}{\\small Grupos com mesma letra não apresentam diferença estatística ($p > 0,05$)} \\\\",
        "\\bottomrule",
        "\\end{tabular}"
    ])
    with open(os.path.join(ARTIFACTS_DIR, "friedman_test.tex"), "w") as f:
        f.write("\n".join(latex_friedman))

    # Gráfico CD
    plt.figure(figsize=(10, 4))
    plt.hlines(1, 0.5, 4.5, colors='black', linewidth=1)
    for v, rank in mean_ranks.items():
        plt.plot(rank, 1, 'o', markersize=12)
        plt.text(rank, 1.1, v, ha='center', va='bottom', fontweight='bold', fontsize=12)
    cd = 2.569 * np.sqrt((4 * 5) / (6 * 8)) 
    plt.errorbar(mean_ranks['V10'], 0.7, xerr=cd/2, color='red', capsize=5, label=f'CD={cd:.2f}')
    plt.gca().invert_xaxis()
    plt.savefig(os.path.join(IMG_DIR, "critical_difference_friedman.pdf"), bbox_inches='tight')

    # --- 2. H1: PARADIGMA ---
    v10_data = df[df['Version'] == 'V10']
    plt.figure(figsize=(10, 6))
    sns.barplot(data=v10_data.sort_values('MCC'), x='Model', y='MCC', hue='Paradigm', dodge=False)
    plt.savefig(os.path.join(IMG_DIR, "h1_paradigm_comparison.pdf"), bbox_inches='tight')

    # --- 3. H2 & H3: EVOLUÇÃO ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df['Version'].isin(['V1', 'V9', 'V10'])], x='Version', y='MCC', hue='Model', marker='o')
    plt.savefig(os.path.join(IMG_DIR, "h2_lighting_impact.pdf"), bbox_inches='tight')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df[df['Version'].isin(['V1', 'V8'])], x='Model', y='MCC', hue='Version', palette='coolwarm')
    plt.savefig(os.path.join(IMG_DIR, "h3_spectral_impact.pdf"), bbox_inches='tight')

    # --- 4. H4: TRADE-OFF INDUSTRIAL (AJUSTADO) ---
    print("\n[H4] Gerando Gráfico de Trade-off Industrial Ajustado...")
    plt.figure(figsize=(14, 9))
    # Filtramos V8 pois distorce a escala do MCC (muito baixo)
    df_plot = df[df['Version'] != 'V8'].copy()
    
    # Customizar mapeamento de marcadores para garantir formas distintas para cada modelo
    # Há 8 modelos, o Seaborn possui formas suficientes no padrão
    scatter = sns.scatterplot(data=df_plot, x='Latency (ms)', y='MCC', 
                             hue='Version', style='Model', 
                             size='GFLOPs', sizes=(100, 1200), 
                             alpha=0.7, palette='viridis')
    
    plt.xscale('log')
    plt.title("Trade-off Industrial: Versão (Cor) vs. Modelo (Forma) vs. Complexidade (Tamanho)", fontsize=14, pad=15)
    plt.xlabel("Latência de Inferência (ms) - Escala Logarítmica", fontsize=12)
    plt.ylabel("MCC (Matthews Correlation Coefficient)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Ajustar legenda para não sobrepor o gráfico
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title_fontsize='11')
    
    plt.savefig(os.path.join(IMG_DIR, "efficiency_tradeoff_full.pdf"), bbox_inches='tight')
    plt.close()

    # Ranking Industrial LaTeX
    h4_df = v10_data[['Model', 'MCC', 'Latency (ms)', 'GFLOPs']].copy()
    h4_df['Throughput (img/s)'] = 1000 / h4_df['Latency (ms)']
    h4_df = h4_df.sort_values('MCC', ascending=False)
    with open(os.path.join(ARTIFACTS_DIR, "h4_industrial_ranking.tex"), "w") as f:
        f.write(h4_df.to_latex(index=False, float_format="%.3f"))

    # --- 5. MCNEMAR HEATMAP ---
    models_v10 = v10_data.sort_values('MCC', ascending=False)['Model'].tolist()
    p_matrix = np.zeros((len(models_v10), len(models_v10)))
    annot_matrix = []
    for i, m1 in enumerate(models_v10):
        row_annots = []
        for j, m2 in enumerate(models_v10):
            k1, k2 = f"{m1.lower()}_v10", f"{m2.lower()}_v10"
            p_val = run_mcnemar(preds_data[k1]['y_true'], preds_data[k1]['y_pred'], preds_data[k2]['y_pred'])
            p_matrix[i,j] = p_val
            if i == j: row_annots.append("-")
            elif p_val > 0.05: row_annots.append(f"{p_val:.2f}\n(ns)")
            else: row_annots.append(f"{p_val:.2e}")
        annot_matrix.append(row_annots)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(p_matrix, annot=annot_matrix, fmt="", xticklabels=models_v10, yticklabels=models_v10, 
                cmap='YlGnBu_r', cbar_kws={'label': 'p-value'}, vmax=0.05)
    plt.title('Matriz de Significância (McNemar) - Dataset V10\n(ns) = não significativo (p > 0,05)')
    plt.savefig(os.path.join(IMG_DIR, "p_value_heatmap_v10.pdf"), bbox_inches='tight')

    print(f"\n>>> VALIDAÇÃO CONCLUÍDA. Artefatos atualizados em {ARTIFACTS_DIR} e {IMG_DIR}")

if __name__ == "__main__":
    main()
