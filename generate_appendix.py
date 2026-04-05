import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import optuna
from optuna.visualization import plot_parallel_coordinate
import plotly.io as pio
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast
import numpy as np

# --- CONFIGURAÇÕES ---
OPTUNA_DB_URI = "sqlite:///optuna_dissertacao.db"
MLFLOW_TRACKING_URI = "file:///Users/rodrigoqaz/Documents/projetos/mecai.dissertacao/mlruns"
PATH_DIR = "tex/appendix/experiments"
OUTPUT_DIR = Path("dissertacao/"+PATH_DIR)
PLOT_FONT_SIZE = 18

# Configurar Plotly para salvar PDF (requer kaleido)
pio.renderers.default = "pdf"

class AppendixGenerator:
    def __init__(self, optuna_uri: str, mlflow_uri: str, output_path: Path):
        self.optuna_uri = optuna_uri
        self.mlflow_uri = mlflow_uri
        self.output_path = output_path
        self.client = MlflowClient(tracking_uri=mlflow_uri)
        mlflow.set_tracking_uri(mlflow_uri)
        
        self.output_path.mkdir(parents=True, exist_ok=True)

    def get_all_experiments(self) -> List[Any]:
        exps = self.client.search_experiments()
        ignore_list = ["Default", "test_experiment"]
        return [e for e in exps if e.name not in ignore_list]

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
        """Plota o histórico de otimização no estilo Optuna usando Matplotlib."""
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
        """Gera uma única imagem PDF com History, Learning Curve e Confusion Matrix."""
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
            print(f"    - Dashboard consolidado gerado (Fonte {PLOT_FONT_SIZE}).")

        except Exception as e:
            print(f"    [ERRO] Falha ao gerar dashboard para {study_name}: {e}")

    def export_parallel_coordinates(self, study_name: str, target_path: Path):
        """Mantém o gráfico de coordenadas paralelas nativo do Optuna em um arquivo separado."""
        try:
            study = optuna.load_study(study_name=study_name, storage=self.optuna_uri)
            complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if not complete_trials: return

            try:
                importance = optuna.importance.get_param_importances(study)
                params_to_plot = list(importance.keys())[:10]
            except:
                all_trial_params = [t.params for t in complete_trials if t.params]
                df_params = pd.DataFrame(all_trial_params)
                params_to_plot = [col for col in df_params.columns if df_params[col].nunique() > 1][:10]
            
            if params_to_plot:
                fig_parallel = plot_parallel_coordinate(study, params=params_to_plot)
                new_dims = []
                annotations = []
                
                if fig_parallel.data and hasattr(fig_parallel.data[0], 'dimensions'):
                    dimensions = fig_parallel.data[0].dimensions
                    num_dims = len(dimensions)
                    for i, dim in enumerate(dimensions):
                        dim_dict = dim.to_plotly_json()
                        truncated_label = dim_dict.get('label', '')
                        full_label = truncated_label
                        if truncated_label != 'Objective Value':
                            if '...' in truncated_label:
                                prefix = truncated_label.replace('...', '')
                                for param_name in params_to_plot:
                                    if param_name.startswith(prefix):
                                        full_label = param_name
                                        break
                            else:
                                for param_name in params_to_plot:
                                    if param_name == truncated_label:
                                        full_label = param_name
                                        break
                        
                        label = full_label
                        if label != 'Objective Value':
                            for prefix in ['convnext_', 'swin_', 'vit_', 'resnet_', 'efficientnet_', 'vgg_', 'inception_', 'densenet_']:
                                if label.startswith(prefix): label = label[len(prefix):]
                            label = label.replace('_', ' ').title()
                            label = label.replace('Learning Rate', 'LR').replace('Architecture', 'Model')
                            label = label.replace('Label Smoothing', 'Smoothing').replace('Weight Decay', 'W. Decay')
                        
                        if len(label) > 10 and ' ' in label:
                            words = label.split(' ')
                            mid = len(words) // 2
                            label = " ".join(words[:mid]) + "<br>" + " ".join(words[mid:])
                        
                        dim_dict['label'] = " " * (i + 1)
                        new_dims.append(dim_dict)
                        x_pos = i / (num_dims - 1) if num_dims > 1 else 0.5
                        annotations.append(dict(
                            x=x_pos, y=-0.08, xref='paper', yref='paper',
                            text=label, showarrow=False, xanchor='center', yanchor='top', font=dict(size=PLOT_FONT_SIZE, color="black")
                        ))
                    
                    fig_parallel.update_traces(dimensions=new_dims)
                    fig_parallel.update_traces(labelfont=dict(color='white', size=1))

                fig_parallel.update_layout(title=None, annotations=annotations, margin=dict(l=60, r=60, t=40, b=160), font=dict(size=PLOT_FONT_SIZE))
                fig_parallel.write_image(str(target_path / "optuna_parallel.pdf"), width=1600, height=800)
                print(f"    - Coordenadas Paralelas exportadas (Fonte {PLOT_FONT_SIZE}).")
        except Exception as e:
            print(f"    [AVISO] Falha em Parallel Coordinates: {e}")

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
                    clean_params['Taxa de Aprendizado (LR)'] = f"{lr:.2e}" if lr < 0.01 else f"{lr:.4f}"
                    clean_params['Decaimento de Peso (WD)'] = f"{wd:.4f}"
            elif k == 'scheduler' and isinstance(v, dict):
                sched_type = v.get('type', 'N/A')
                clean_params['Agendador de LR'] = sched_type
                if 'params' in v:
                    if sched_type == 'CosineAnnealingWarmRestarts': clean_params['Agendador (MARKERTZERO)'] = str(v['params'].get('T_0', ''))
                    elif sched_type == 'ReduceLROnPlateau': clean_params['Agendador (Patience)'] = str(v['params'].get('patience', ''))
            elif k == 'loss' and isinstance(v, dict):
                clean_params['Função de Perda'] = v.get('type', 'CrossEntropyLoss')
                if 'params' in v and 'label_smoothing' in v['params']: clean_params['Label Smoothing'] = f"{v['params']['label_smoothing']:.4f}"
            elif k in ['augmentation config', 'augmentation_config'] and isinstance(v, dict):
                if v.get('use_augmentation'):
                    clean_params['Augmentation (Método)'] = str(v.get('per_image_aug_method', '')).title()
                    if 'basic_params' in v:
                        bp = v['basic_params']
                        clean_params['Aug (Rotação)'] = f"{bp.get('rotation_range', 0)}MARKERDEGREE"
                        clean_params['Aug (Prob. Espelhamento)'] = f"{bp.get('horizontal_flip_prob', 0):.2f}"
                        clean_params['Aug (Brilho / Contraste)'] = f"{bp.get('brightness_range', 0):.2f} / {bp.get('contrast_range', 0):.2f}"
                else: clean_params['Data Augmentation'] = "False"
            else:
                key_name = str(k).replace('_', ' ').title()
                clean_params[key_name] = f"{v:.4f}" if isinstance(v, float) else str(v)
        
        left_rows = []
        for key in sorted(clean_params.keys()):
            val = clean_params[key]
            safe_k = self.latex_escape(key).replace('MARKERTZERO', '$T_0$')
            safe_v = self.latex_escape(str(val)).replace('MARKERDEGREE', '$^\\circ$')
            left_rows.append(f"{safe_k} & {safe_v}")

        right_rows = []
        for cls in classes:
            f1, prec, rec = metrics.get(f"class/f1_{cls}", 0), metrics.get(f"class/precision_{cls}", 0), metrics.get(f"class/recall_{cls}", 0)
            right_rows.append(f"{self.latex_escape(cls)} & {prec:.4f} & {rec:.4f} & {f1:.4f}")

        macro_prec, macro_rec, macro_f1, mcc = metrics.get('final/macro avg_precision', 0), metrics.get('final/macro avg_recall', 0), metrics.get('final/macro avg_f1-score', 0), metrics.get('final/matthews_corrcoef', 0)
        right_rows.append(f"\\textbf{{Média (Macro)}} & {macro_prec:.4f} & {macro_rec:.4f} & {macro_f1:.4f}")
        right_rows.append(f"\\textbf{{MCC}} & \\multicolumn{{3}}{{c}}{{{mcc:.4f}}}")

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
        \includegraphics[width=0.9\textwidth]{<PATH_DIR>/<RAW_EXP_NAME>/dashboard_consolidado.pdf}
        \caption{Histórico de Otimização, Curvas de Aprendizado e Matriz de Confusão - <EXP_NAME>}
    \end{figure}
\end{minipage}

\clearpage
\pdfpagewidth=297mm \pdfpageheight=210mm
\newgeometry{left=1.5cm, right=1.5cm, top=2.5cm, bottom=2cm} 
\begin{figure}[H]
    \centering
    \makebox[\textwidth][l]{\includegraphics[width=1.7\textwidth]{<PATH_DIR>/<RAW_EXP_NAME>/optuna_parallel.pdf}}
    \caption{Coordenadas Paralelas dos principais hiperparâmetros para <EXP_NAME>}
\end{figure}
\clearpage
\pdfpagewidth=210mm \pdfpageheight=297mm
\restoregeometry
\newpage
"""
        content = latex_template.replace("<EXP_NAME>", safe_exp_name).replace("<RAW_EXP_NAME>", exp_name).replace("<COMBINED_ROWS>", combined_rows).replace("<PATH_DIR>", PATH_DIR)
        with open(target_path / "relatorio_experimento.tex", "w") as f: f.write(content)
        print(f"    - Snippet LaTeX gerado (Tabela e Dashboard na mesma página).")

    def run(self):
        experiments = self.get_all_experiments()
        print(f"Encontrados {len(experiments)} experimentos para processar.")
        for exp in experiments:
            print(f"\n>>> Processando: {exp.name}")
            exp_dir = self.output_path / exp.name
            exp_dir.mkdir(exist_ok=True)
            champion = self.find_champion_run(exp.experiment_id)
            if not champion: continue
            
            self.generate_combined_dashboard(exp.name, champion.info.run_id, exp_dir)
            self.export_parallel_coordinates(exp.name, exp_dir)
            
            classes = []
            try:
                local_path = self.client.download_artifacts(champion.info.run_id, "confusion_matrix.csv", str(exp_dir))
                classes = pd.read_csv(local_path, index_col=0).columns.tolist()
            except: pass
            
            self.generate_latex_snippet(exp.name, champion, classes, exp_dir)

if __name__ == "__main__":
    generator = AppendixGenerator(OPTUNA_DB_URI, MLFLOW_TRACKING_URI, OUTPUT_DIR)
    generator.run()
    print("\nProcesso concluído!")
