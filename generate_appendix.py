import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import optuna
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import numpy as np

from src.utils.latex_fmt import br_num

# --- CONFIGURAÇÕES ---
OPTUNA_DB_URI = "sqlite:///optuna_dissertacao.db"
MLFLOW_TRACKING_URI = "file:///Users/rodrigoqaz/Documents/projetos/mecai.dissertacao/mlruns"
PATH_DIR = "tex/appendix/experiments"
OUTPUT_DIR = Path("dissertacao/"+PATH_DIR)
INDEX_FILE_PATH = Path("dissertacao/tex/appendix/appendice_2.tex")
PLOT_FONT_SIZE = 18

class AppendixGenerator:
    def __init__(self, optuna_uri: str, mlflow_uri: str, output_path: Path):
        self.optuna_uri = optuna_uri
        self.mlflow_uri = mlflow_uri
        self.output_path = output_path
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.processed_experiments = []

    def get_all_experiments(self) -> List[Any]:
        exps = self.client.search_experiments()
        # "*_Optimization" são execuções exploratórias antigas, anteriores à varredura
        # sistemática v1-v12 (MCC=0, sem confusion_matrix/dashboard na maioria) - nunca
        # fizeram parte do índice do Apêndice B e não devem ser regeneradas.
        ignore_list = [
            "Default", "test_experiment",
            "EfficientNet_Optimization", "Inception_Optimization",
            "ResNet_Optimization", "VGGNet_Optimization",
        ]
        # Ordenar alfabeticamente pelo nome para consistência no índice
        return sorted([e for e in exps if e.name not in ignore_list], key=lambda x: x.name)

    def find_champion_run(self, experiment_id: str) -> Optional[Any]:
        try:
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["metrics.\"final/matthews_corrcoef\" DESC", "metrics.\"val/loss\" ASC"],
                max_results=1
            )
            return runs[0] if runs else None
        except Exception as e:
            print(f"    [ERRO] Falha ao buscar champion run: {e}")
            return None

    def plot_optimization_history_matplotlib(self, study: optuna.study.Study, ax: plt.Axes):
        trials = study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            ax.text(0.5, 0.5, "Sem trials completados", ha='center', fontsize=PLOT_FONT_SIZE)
            return

        trial_numbers = [t.number for t in completed_trials]
        values = [t.value for t in completed_trials]
        
        best_values = []
        current_best = -np.inf if study.direction == optuna.study.StudyDirection.MAXIMIZE else np.inf
        for v in values:
            if study.direction == optuna.study.StudyDirection.MAXIMIZE:
                current_best = max(current_best, v)
            else:
                current_best = min(current_best, v)
            best_values.append(current_best)

        ax.scatter(trial_numbers, values, color='blue', alpha=0.6, label='Objetivo (Trial)', s=50)
        ax.plot(trial_numbers, best_values, color='red', marker='o', markersize=6, label='Melhor Valor', linewidth=2)
        
        ax.set_title("Histórico de Otimização", fontsize=PLOT_FONT_SIZE, fontweight='bold')
        ax.set_xlabel("Número do Trial", fontsize=PLOT_FONT_SIZE)
        ax.set_ylabel("MCC", fontsize=PLOT_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE)
        ax.legend(loc='best', fontsize=PLOT_FONT_SIZE)
        ax.grid(True, linestyle='--', alpha=0.6)

    def generate_combined_dashboard(self, study_name: str, run_id: str, target_path: Path):
        try:
            study = optuna.load_study(study_name=study_name, storage=self.optuna_uri)
            train_loss = self.client.get_metric_history(run_id, "train/loss")
            val_loss = self.client.get_metric_history(run_id, "val/loss")
            cm_path = self.client.download_artifacts(run_id, "confusion_matrix.csv", str(target_path))
            df_cm = pd.read_csv(cm_path, index_col=0)

            fig = plt.figure(figsize=(20, 14))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.5])

            ax_hist = fig.add_subplot(gs[0, :])
            self.plot_optimization_history_matplotlib(study, ax_hist)

            ax_curve = fig.add_subplot(gs[1, 0])
            if train_loss:
                ax_curve.plot([m.step for m in train_loss], [m.value for m in train_loss], label='Treino', linewidth=3)
            if val_loss:
                ax_curve.plot([m.step for m in val_loss], [m.value for m in val_loss], label='Validação', linewidth=3)
            ax_curve.set_title("Curva de Aprendizado (Loss)", fontsize=PLOT_FONT_SIZE, fontweight='bold')
            ax_curve.set_xlabel("Épocas", fontsize=PLOT_FONT_SIZE)
            ax_curve.set_ylabel("Loss", fontsize=PLOT_FONT_SIZE)
            ax_curve.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE)
            ax_curve.legend(fontsize=PLOT_FONT_SIZE)
            ax_curve.grid(True, linestyle='--', alpha=0.6)

            ax_cm = fig.add_subplot(gs[1, 1])
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": PLOT_FONT_SIZE})
            ax_cm.set_title("Matriz de Confusão", fontsize=PLOT_FONT_SIZE, fontweight='bold')
            ax_cm.set_ylabel("Real", fontsize=PLOT_FONT_SIZE)
            ax_cm.set_xlabel("Predito", fontsize=PLOT_FONT_SIZE)
            ax_cm.tick_params(axis='both', which='major', labelsize=PLOT_FONT_SIZE)

            plt.tight_layout(pad=4.0)
            plt.savefig(target_path / "dashboard_consolidado.pdf", bbox_inches='tight')
            plt.close()
            print(f"    - Dashboard consolidado gerado.")

        except Exception as e:
            print(f"    [ERRO] Falha ao gerar dashboard para {study_name}: {e}")

    def latex_escape(self, text: str) -> str:
        if not isinstance(text, str): text = str(text)
        chars = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
        return "".join(chars.get(c, c) for c in text)

    def generate_latex_snippet(self, exp_name: str, run_data: Any, classes: List[str], target_path: Path):
        params = run_data.data.params
        metrics = run_data.data.metrics
        safe_exp_name = self.latex_escape(exp_name)

        blacklist = {'callbacks', 'data dir', 'data_dir', 'dataset name', 'dataset_name', 'device', 'experiment name', 'experiment_name', 'global config', 'global_config', 'mlflow uri', 'mlflow_uri', 'num workers', 'num_workers', 'params', 'weights', 'augmentations', 'pretrained', 'model type', 'model_type', 'max epochs', 'max_epochs', 'seed', 'input channels', 'input_channels', 'num classes', 'num_classes'}
        clean_params = {}

        for k, v_raw in params.items():
            if k in blacklist: continue
            v = v_raw
            if isinstance(v_raw, str):
                try: v = ast.literal_eval(v_raw)
                except: pass

            if k == 'optimizer' and isinstance(v, dict):
                clean_params['Otimizador'] = v.get('type', 'AdamW')
                if 'params' in v:
                    lr, wd = v['params'].get('lr', 0.0), v['params'].get('weight_decay', 0.0)
                    # Notação exponencial em texto simples (sem \times LaTeX): o valor passa
                    # por latex_escape() logo abaixo, que mutilaria comandos LaTeX crus.
                    clean_params['Taxa de Aprendizado (LR)'] = f"{lr:.2e}".replace(".", ",") if lr < 0.01 else br_num(lr, 4)
                    clean_params['Decaimento de Peso (WD)'] = br_num(wd, 4)
            elif k == 'scheduler' and isinstance(v, dict):
                sched_type = v.get('type', 'N/A')
                clean_params['Agendador de LR'] = sched_type
                if 'params' in v:
                    if sched_type == 'CosineAnnealingWarmRestarts': clean_params['Agendador ($T_0$)'] = str(v['params'].get('T_0', ''))
                    elif sched_type == 'ReduceLROnPlateau': clean_params['Agendador (Patience)'] = str(v['params'].get('patience', ''))
            elif k == 'loss' and isinstance(v, dict):
                clean_params['Função de Perda'] = v.get('type', 'CrossEntropyLoss')
                if 'params' in v and 'label_smoothing' in v['params']: clean_params['Label Smoothing'] = br_num(v['params']['label_smoothing'], 4)
            elif k in ['augmentation config', 'augmentation_config'] and isinstance(v, dict):
                if v.get('use_augmentation'):
                    clean_params['Augmentation (Método)'] = str(v.get('per_image_aug_method', '')).title()
                    if 'basic_params' in v:
                        bp = v['basic_params']
                        clean_params['Aug (Rotação)'] = f"{bp.get('rotation_range', 0)}$^\\circ$"
                        clean_params['Aug (Prob. Espelhamento)'] = br_num(bp.get('horizontal_flip_prob', 0), 2)
                        clean_params['Aug (Brilho / Contraste)'] = f"{br_num(bp.get('brightness_range', 0), 2)} / {br_num(bp.get('contrast_range', 0), 2)}"
                else: clean_params['Data Augmentation'] = "False"
            else:
                key_name = str(k).replace('_', ' ').title()
                clean_params[key_name] = br_num(v, 4) if isinstance(v, float) else str(v)
        
        left_rows = []
        for key in sorted(clean_params.keys()):
            val = clean_params[key]
            safe_k = self.latex_escape(key)
            safe_v = self.latex_escape(str(val))
            left_rows.append(f"{safe_k} & {safe_v}")

        right_rows = []
        for cls in classes:
            f1, prec, rec = metrics.get(f"class/f1_{cls}", 0), metrics.get(f"class/precision_{cls}", 0), metrics.get(f"class/recall_{cls}", 0)
            right_rows.append(f"{self.latex_escape(cls)} & {br_num(prec, 4)} & {br_num(rec, 4)} & {br_num(f1, 4)}")

        macro_prec, macro_rec, macro_f1, mcc = metrics.get('final/macro avg_precision', 0), metrics.get('final/macro avg_recall', 0), metrics.get('final/macro avg_f1-score', 0), metrics.get('final/matthews_corrcoef', 0)
        right_rows.append(f"\\textbf{{Média (Macro)}} & {br_num(macro_prec, 4)} & {br_num(macro_rec, 4)} & {br_num(macro_f1, 4)}")
        right_rows.append(f"\\textbf{{MCC}} & \\multicolumn{{3}}{{c}}{{{br_num(mcc, 4)}}}")

        max_len = max(len(left_rows), len(right_rows))
        combined_rows = ""
        for i in range(max_len):
            l = left_rows[i] if i < len(left_rows) else " & "
            r = right_rows[i] if i < len(right_rows) else " & & & "
            if i == len(classes): combined_rows += "\\cline{3-6} \n"
            combined_rows += f"{l} & {r} \\\\ \n"

        latex_template = r"""
\noindent\begin{minipage}{\textwidth}
    \begin{table}[H]
        \centering
        \caption{Hiperparâmetros e Resultados do Melhor Modelo - <EXP_NAME>}
        \vspace{0.2cm}
        \resizebox{\textwidth}{!}{%
            \begin{tabular}{ll | lrrr}
                \hline
                \multicolumn{2}{c|}{\textbf{Hiperparâmetros}} & \multicolumn{4}{c}{\textbf{Métricas por Classe}} \\ \hline
                \textbf{Parâmetro} & \textbf{Valor} & \textbf{Classe} & \textbf{Precisão} & \textbf{Recall} & \textbf{F1-Score} \\ \hline
    <COMBINED_ROWS>            \hline
            \end{tabular}%
        }
    \end{table}

    \begin{figure}[H]
        \centering
        \includegraphics[width=0.85\textwidth]{<PATH_DIR>/<RAW_EXP_NAME>/dashboard_consolidado.pdf}
        \caption{Histórico de Otimização, Curvas de Aprendizado e Matriz de Confusão - <EXP_NAME>}
    \end{figure}
\end{minipage}
\clearpage
"""
        content = latex_template.replace("<EXP_NAME>", safe_exp_name).replace("<RAW_EXP_NAME>", exp_name).replace("<COMBINED_ROWS>", combined_rows).replace("<PATH_DIR>", PATH_DIR)
        with open(target_path / "relatorio_experimento.tex", "w") as f: f.write(content)
        print(f"    - Snippet LaTeX gerado.")

    def generate_index_file(self):
        print(f"\n>>> Gerando arquivo de índice: {INDEX_FILE_PATH}")
        # Decisão editorial: os relatórios individuais (curvas de convergência, métricas por
        # época, logs do MLflow) ficam de fora do documento impresso, por extensão, e são
        # apenas referenciados no repositório oficial do projeto. Os \input ficam comentados
        # abaixo (não removidos) para permitir reativação pontual se algum relatório for citado
        # explicitamente no texto; os arquivos continuam sendo gerados/versionados normalmente.
        content = r"""\chapter{Experimentos}
\label{apendice:experimentos}

Este apêndice apresenta os detalhes técnicos, hiperparâmetros otimizados e resultados de desempenho para cada experimento realizado durante a fase de busca bayesiana. Cada página detalha o comportamento de uma arquitetura sob uma configuração específica de conjunto de dados. O Coeficiente de Correlação de Matthews (MCC) é adotado como métrica principal de desempenho nos relatórios.

Devido à extensão e ao detalhamento técnico dos 52 experimentos realizados nesta pesquisa (abrangendo diversas combinações de arquiteturas, condições de iluminação e técnicas de aumento de dados), os relatórios individuais de treinamento, contendo curvas de convergência, métricas por época e logs detalhados do MLflow, foram movidos para o repositório oficial do projeto.

Esta decisão visa manter a concisão deste documento, priorizando a análise dos modelos de melhor desempenho discutidos nos Capítulos 4 e 5. Os relatórios completos de cada \textit{trial} da otimização bayesiana podem ser consultados e auditados no seguinte endereço:

\begin{center}
    \url{https://github.com/rodrigoqaz/mecai.dissertacao/tree/main/results/analysis}
\end{center}


% \clearpage
"""
        for exp_name in self.processed_experiments:
            clean_name = exp_name.replace('_', ' ').title()
            content += f"\n% \\section*{{{clean_name}}}\n"
            content += f"%     \\input{{tex/appendix/experiments/{exp_name}/relatorio_experimento.tex}}\n"

        with open(INDEX_FILE_PATH, "w") as f:
            f.write(content)
        print("    - Arquivo de índice gerado com sucesso.")

    def run(self):
        experiments = self.get_all_experiments()
        print(f"Encontrados {len(experiments)} experimentos para processar.")
        for exp in experiments:
            print(f"\n>>> Processando: {exp.name}")
            exp_dir = self.output_path / exp.name
            exp_dir.mkdir(parents=True, exist_ok=True)
            champion = self.find_champion_run(exp.experiment_id)
            if not champion: continue
            
            self.generate_combined_dashboard(exp.name, champion.info.run_id, exp_dir)
            
            classes = []
            try:
                local_path = self.client.download_artifacts(champion.info.run_id, "confusion_matrix.csv", str(exp_dir))
                classes = pd.read_csv(local_path, index_col=0).columns.tolist()
            except: pass
            
            self.generate_latex_snippet(exp.name, champion, classes, exp_dir)
            self.processed_experiments.append(exp.name)
        
        self.generate_index_file()

if __name__ == "__main__":
    generator = AppendixGenerator(OPTUNA_DB_URI, MLFLOW_TRACKING_URI, OUTPUT_DIR)
    generator.run()
    print("\nProcesso concluído!")
