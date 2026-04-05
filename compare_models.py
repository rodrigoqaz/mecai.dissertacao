import os
import time
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    cohen_kappa_score, matthews_corrcoef, accuracy_score
)
from src.data.data_loader import load_datasets
from models.model_factory import ModelFactory
from src.config.models.config import Config
from typing import Dict, List, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/comparison"
MLRUNS_DIR = "mlruns"
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_best_mlflow_run(model_type: str, dataset_name: str = "v8") -> Optional[Dict]:
    """
    Busca na pasta mlruns o melhor run para um dado model_type e dataset_name.
    Retorna um dicionário com o caminho dos pesos e a configuração.
    """
    best_acc = -1.0
    best_run_info = None

    if not os.path.exists(MLRUNS_DIR):
        return None

    # Itera pelos experimentos
    for exp_id in os.listdir(MLRUNS_DIR):
        exp_path = os.path.join(MLRUNS_DIR, exp_id)
        if not os.path.isdir(exp_path) or exp_id == "models" or exp_id == ".trash":
            continue
        
        # Itera pelos runs
        for run_id in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run_id)
            params_path = os.path.join(run_path, "params")
            metrics_path = os.path.join(run_path, "metrics")
            
            if not os.path.exists(params_path) or not os.path.exists(metrics_path):
                continue
            
            # Verifica model_type e dataset_name
            try:
                with open(os.path.join(params_path, "model_type"), "r") as f:
                    m_type = f.read().strip()
                with open(os.path.join(params_path, "dataset_name"), "r") as f:
                    d_name = f.read().strip()
                
                if m_type != model_type or d_name != dataset_name:
                    continue
                
                # Pega a melhor acurácia
                acc = 0.0
                acc_file = os.path.join(metrics_path, "val/accuracy") # Tenta métrica de época
                if not os.path.exists(acc_file):
                    acc_file = os.path.join(metrics_path, "val/final_accuracy") # Tenta métrica final
                
                if os.path.exists(acc_file):
                    with open(acc_file, "r") as f:
                        last_line = f.readlines()[-1]
                        acc = float(last_line.split()[1])
                
                if acc > best_acc:
                    # Verifica se o modelo existe em artifacts
                    # O MLflow salva em artifacts/best_model/data/model.pth (via log_model)
                    # Ou em artifacts/final_artifacts/best_model_*.pth (se foi via log_artifact)
                    potential_weights = [
                        os.path.join(run_path, "artifacts", "best_model", "data", "model.pth"),
                        os.path.join(run_path, "artifacts", "final_model", "data", "model.pth")
                    ]
                    
                    # Também busca por arquivos .pth soltos em artifacts
                    artifacts_root = os.path.join(run_path, "artifacts")
                    if os.path.exists(artifacts_root):
                        for root, _, files in os.walk(artifacts_root):
                            for f in files:
                                if f.endswith(".pth"):
                                    potential_weights.append(os.path.join(root, f))

                    valid_weight = None
                    for w in potential_weights:
                        if os.path.exists(w):
                            valid_weight = w
                            break
                    
                    if valid_weight:
                        best_acc = acc
                        best_run_info = {
                            'run_id': run_id,
                            'weights': valid_weight,
                            'accuracy': acc,
                            'config': f"best_config_{model_type}_{model_type}_v8_AB_opt_aug.yaml" # Fallback config
                        }
            except Exception:
                continue
                
    return best_run_info

def load_model_from_config(config_path: str, weights_path: str, num_classes: int, input_channels: int):
    # Se o arquivo de configuração não existir, tentamos carregar um padrão para o modelo
    if not os.path.exists(config_path):
        model_type = os.path.basename(config_path).split('_')[2]
        print(f"Config {config_path} não encontrada. Usando padrão para {model_type}.")
        config_obj = Config.load_model_config(model_type)
    else:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config_obj = type('ModelConfig', (), config_dict)
    
    model, _, _, _, _ = ModelFactory.create_pipeline(
        DEVICE, config_obj, num_classes, input_channels
    )
    
    try:
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        print(f"Erro ao ler arquivo de pesos: {e}")
        raise e
    
    # MLflow log_model salva o state_dict dentro de um wrapper às vezes
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif not isinstance(state_dict, dict):
        # Se carregou o objeto do modelo inteiro (mlflow.pytorch.log_model faz isso)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        else:
            print(f"Aviso: Formato de checkpoint inesperado para {weights_path}")
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def evaluate_model(model, dataloader, class_names):
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Avaliando"):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    total_time = time.time() - start_time
    latency = (total_time / len(dataloader.dataset)) * 1000 
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'kappa': cohen_kappa_score(all_labels, all_preds),
        'latency_ms': latency,
        'params_count': sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6 
    }
    
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    metrics['f1_macro'] = report['macro avg']['f1-score']
    
    return metrics, all_labels, all_preds

def main():
    model_types = ['resnet', 'inception', 'efficientnet', 'densenet', 'vgg', 'vit', 'swin', 'convnext']
    dataset_version = "v8"
    
    # Setup Data
    base_config = Config.load_model_config('resnet')
    base_config.data_dir = f"data/gold/datasets/{dataset_version}/train_val"
    generator = torch.Generator().manual_seed(42)
    _, val_loader, num_classes, input_channels, _, class_names, _ = load_datasets(
        batch_size=32, config=base_config, generator=generator
    )
    
    results = []
    
    for m_type in model_types:
        print(f"\n>>> Buscando melhor run para {m_type} (Dataset {dataset_version})...")
        info = find_best_mlflow_run(m_type, dataset_version)
        
        if not info:
            print(f"Nenhum run v8 encontrado para {m_type}.")
            continue
            
        print(f"Melhor run encontrado: {info['run_id']} (Acc: {info['accuracy']:.4f})")
        print(f"Caminho: {info['weights']}")
        
        try:
            model = load_model_from_config(info['config'], info['weights'], num_classes, input_channels)
            metrics, labels, preds = evaluate_model(model, val_loader, class_names)
            
            results.append({
                'Modelo': m_type.upper(),
                'Acurácia': metrics['accuracy'],
                'F1-Macro': metrics['f1_macro'],
                'MCC': metrics['mcc'],
                'Kappa': metrics['kappa'],
                'Latência (ms)': metrics['latency_ms'],
                'Params (M)': metrics['params_count'],
                'Run_ID': info['run_id']
            })
            
            # Matriz de Confusão
            cm = confusion_matrix(labels, preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Matriz de Confusão - {m_type.upper()}')
            plt.savefig(os.path.join(RESULTS_DIR, f'cm_{m_type}.png'))
            plt.close()

        except Exception as e:
            print(f"Erro ao avaliar {m_type}: {e}")
            continue
            
    if results:
        df = pd.DataFrame(results).sort_values(by='Acurácia', ascending=False)
        df.to_csv(os.path.join(RESULTS_DIR, "comparacao_modelos_mlflow.csv"), index=False)
        print("\n" + "="*50)
        print("RESULTADOS FINAIS (DADOS MLFLOW)")
        print("="*50)
        print(df.drop(columns=['Run_ID']).to_string(index=False))
        
        # Tabela LaTeX
        latex = r"\begin{table}[h]" + "\n" + r"\centering" + "\n" + r"\begin{tabular}{lcccccc}" + "\n" + r"\hline" + "\n"
        latex += r"Modelo & Acurácia & F1-Macro & MCC & Kappa & Latência (ms) & Params (M) \\ \hline" + "\n"
        for _, row in df.iterrows():
            latex += f"{row['Modelo']} & {row['Acurácia']:.4f} & {row['F1-Macro']:.4f} & {row['MCC']:.4f} & {row['Kappa']:.4f} & {row['Latência (ms)']:.2f} & {row['Params (M)']:.2f} \\\\" + "\n"
        latex += r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"
        with open(os.path.join(RESULTS_DIR, "tabela_comparativa.tex"), "w") as f:
            f.write(latex)
        print(f"\nResultados salvos em {RESULTS_DIR}")
    else:
        print("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()
