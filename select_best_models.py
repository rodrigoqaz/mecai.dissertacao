import os
import pandas as pd
import numpy as np
import mlflow
import shutil
import yaml
import ast
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configurações do ambiente
MLRUNS_DIR = "mlruns"
RESULTS_DIR = "results"
MANIFEST_PATH = os.path.join(RESULTS_DIR, "best_models_manifest.csv")
VERSIONS_TO_ANALYZE = ["v1", "v8", "v9", "v10", "v11", "v12"]
MODEL_TYPES = ["resnet", "inception", "efficientnet", "densenet", "vgg", "vit", "swin", "convnext"]

mlflow.set_tracking_uri(f"file://{os.path.abspath(MLRUNS_DIR)}")

def cleanup_inferior_weights(run_id: str, artifact_uri: str):
    """
    Remove arquivos .pth de runs que não foram selecionados como os melhores.
    Mantém os metadados do MLflow intactos, mas limpa o espaço em disco.
    """
    artifact_path = artifact_uri.replace("file://", "")
    if not os.path.exists(artifact_path):
        return

    deleted_count = 0
    for root, _, files in os.walk(artifact_path):
        for f in files:
            if f.endswith(".pth"):
                file_to_delete = os.path.join(root, f)
                try:
                    os.remove(file_to_delete)
                    deleted_count += 1
                except Exception as e:
                    print(f"      [!] Falha ao deletar {f}: {e}")
    
    if deleted_count > 0:
        print(f"      [Limpeza] {deleted_count} arquivos .pth removidos do run {run_id}")

def find_best_runs_and_cleanup() -> List[Dict]:
    """
    Minera o MLflow para encontrar os melhores runs e limpa os pesos dos runs inferiores.
    """
    print("Iniciando mineração e limpeza no MLflow (Critério: MCC)...")
    all_experiments = mlflow.search_experiments()
    exp_ids = [exp.experiment_id for exp in all_experiments]
    
    best_runs_list = []

    for version in VERSIONS_TO_ANALYZE:
        print(f"\n>>> Processando Versão: {version.upper()}...")
        for model in MODEL_TYPES:
            query = f"params.dataset_name = '{version}' and params.model_type = '{model}' and status = 'FINISHED'"
            
            try:
                # Buscamos TODOS os runs para poder limpar os inferiores
                runs = mlflow.search_runs(
                    experiment_ids=exp_ids,
                    filter_string=query,
                    order_by=["metrics.`final/matthews_corrcoef` DESC", "metrics.`val/accuracy` DESC"]
                )
            except Exception as e:
                print(f"  [ERRO] Falha ao buscar {model} ({version}): {e}")
                continue
            
            if runs.empty:
                continue

            # O primeiro é o melhor absoluto
            best_run = runs.iloc[0]
            best_run_id = best_run['run_id']
            
            # Captura métricas
            mcc = best_run.get('metrics.final/matthews_corrcoef', 0.0)
            if pd.isna(mcc): mcc = 0.0
            acc = best_run.get('metrics.val/accuracy', 0.0)
            if pd.isna(acc): acc = 0.0
            
            print(f"  [MELHOR] {model.upper()}: MCC={mcc:.4f} | Run={best_run_id}")
            
            best_runs_list.append({
                'model_type': model,
                'dataset_version': version,
                'run_id': best_run_id,
                'mcc_val': mcc,
                'val_acc': acc,
                'experiment_id': best_run['experiment_id'],
                'artifact_uri': best_run['artifact_uri']
            })

            # Limpeza dos outros runs (perdedores)
            if len(runs) > 1:
                inferior_runs = runs.iloc[1:]
                for _, inf_run in inferior_runs.iterrows():
                    cleanup_inferior_weights(inf_run['run_id'], inf_run['artifact_uri'])

    return best_runs_list

def verify_and_prepare_weights(run_info: Dict) -> Tuple[Optional[str], bool]:
    """
    Verifica a existência física do arquivo .pth e o copia para a pasta de resultados.
    """
    version = run_info['dataset_version']
    model_type = run_info['model_type']
    artifact_root = run_info['artifact_uri'].replace("file://", "")
    
    potential_paths = [
        os.path.join(artifact_root, "best_model", "data", "model.pth"),
        os.path.join(artifact_root, "final_model", "data", "model.pth"),
        os.path.join(artifact_root, "final_artifacts", f"best_model_{model_type}.pth"),
    ]
    
    valid_weight_source = None
    for p in potential_paths:
        if os.path.exists(p):
            valid_weight_source = p
            break
            
    if not valid_weight_source and os.path.exists(artifact_root):
        for root, _, files in os.walk(artifact_root):
            for f in files:
                if f.endswith(".pth") and ("best" in f.lower() or "final" in f.lower()):
                    valid_weight_source = os.path.join(root, f)
                    break
            if valid_weight_source: break

    if valid_weight_source:
        dest_dir = os.path.join(RESULTS_DIR, "weights", version)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{model_type}_best.pth")
        shutil.copy2(valid_weight_source, dest_path)
        return dest_path, False
    else:
        return None, True

def generate_recovery_yaml(run_id: str, model_type: str, version: str) -> str:
    """
    Gera um arquivo YAML de configuração para documentação e retreinamento (se necessário).
    """
    run = mlflow.get_run(run_id)
    params = run.data.params
    
    aug_config_str = params.get('augmentation_config', '{}')
    try:
        aug_config = ast.literal_eval(aug_config_str)
    except:
        aug_config = {"use_augmentation": False}

    config_dict = {
        'model_type': model_type,
        'global_config': {
            'dataset_name': version,
            'batch_size': int(params.get('batch_size', 32)),
            'max_epochs': int(params.get('max_epochs', 100)),
            'device': 'auto',
            'seed': int(params.get('seed', 42)),
            'num_workers': 4,
            'mlflow_uri': f"file://{os.path.abspath(MLRUNS_DIR)}",
            'experiment_name': f"ELITE_EVAL_{version}_{model_type}"
        },
        'augmentation_config': aug_config,
        'params': {k: v for k, v in params.items() if k not in ['dataset_name', 'model_type', 'batch_size', 'max_epochs', 'seed', 'augmentation_config']}
    }

    config_dir = "src/config/models/optimized"
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"best_config_{model_type}_{version}.yaml")
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    return config_path

def main():
    best_runs = find_best_runs_and_cleanup()
    
    if not best_runs:
        print("\n[ERRO] Nenhum run compatível encontrado no MLflow.")
        return

    manifest_data = []

    print(f"\n>>> Verificando e Preparando Artefatos para {len(best_runs)} melhores modelos...")

    for run in best_runs:
        weight_path, needs_recovery = verify_and_prepare_weights(run)
        config_path = generate_recovery_yaml(run['run_id'], run['model_type'], run['dataset_version'])
        
        status = "RECOVERY" if needs_recovery else "READY"
        print(f"  [{status}] {run['model_type'].upper()} ({run['dataset_version']})")

        manifest_data.append({
            'model_type': run['model_type'],
            'dataset_version': run['dataset_version'],
            'run_id': run['run_id'],
            'mcc_val': run['mcc_val'],
            'val_acc': run['val_acc'],
            'weights_path': weight_path if weight_path else "MISSING",
            'needs_retrain': needs_recovery,
            'config_path': config_path
        })

    df = pd.DataFrame(manifest_data)
    df.to_csv(MANIFEST_PATH, index=False)
    
    print(f"\n" + "="*50)
    print(f"FASE 1 CONCLUÍDA (BUSCA, LIMPEZA E MANIFESTO)")
    print(f"="*50)
    print(f"Manifesto: {MANIFEST_PATH}")
    print(f"Modelos Prontos: {len(df[~df['needs_retrain']])}")
    print(f"Modelos para Retreino: {len(df[df['needs_retrain']])}")
    print(f"="*50)

if __name__ == "__main__":
    main()
