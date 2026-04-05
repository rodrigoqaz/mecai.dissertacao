import os
import time
import torch
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
from src.config.models.config import Config
from models.model_factory import ModelFactory
from src.data.data_loader import NPYFolderDataset, compute_stats
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from typing import Dict, List, Tuple

# Configurações de ambiente
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
MANIFEST_PATH = "results/best_models_manifest.csv"
PREDICTIONS_DIR = "results/predictions"
SUMMARY_PATH = "results/evaluation_summary.csv"

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def calcular_complexidade_modelo(model, num_channels=3, image_size=224, device='cpu'):
    """
    Calcula os GFLOPs e o número de parâmetros do modelo usando fvcore.
    """
    model.eval()
    tensor_teste = torch.randn(1, num_channels, image_size, image_size).to(device)
    
    # GFLOPs
    flops_analyzer = FlopCountAnalysis(model, tensor_teste)
    flops_analyzer.unsupported_ops_warnings(False)
    total_flops = flops_analyzer.total()
    gflops = total_flops / 1e9
    
    # Parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    params_millions = total_params / 1e6
    
    return gflops, params_millions

def get_test_loader(version: str, input_channels: int, batch_size: int = 1):
    train_val_dir = f"data/gold/datasets/{version}/train_val"
    test_dir = f"data/gold/datasets/{version}/test"

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Pasta de teste não encontrada: {test_dir}")

    stats_dataset = NPYFolderDataset(root_dir=train_val_dir)
    stats_loader = DataLoader(stats_dataset, batch_size=32, shuffle=False, num_workers=0)
    norm_mean, norm_std = compute_stats(stats_loader, stats_dataset)

    base_transforms = v2.Compose([
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std)
    ])

    dataset = NPYFolderDataset(root_dir=test_dir, transform=base_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, dataset.classes

def run_benchmarking(model, loader, device) -> Dict:
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    latencies = []

    print(f"  Benchmarking on {device}...")
    
    with torch.no_grad():
        sample_input, _ = loader.dataset[0]
        dummy_input = sample_input.unsqueeze(0).to(device)
        for _ in range(5): _ = model(dummy_input)

        for inputs, labels in tqdm(loader, desc="  Inferência", leave=False):
            inputs = inputs.to(device)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            elif device.type == 'mps': torch.mps.synchronize()
            
            start_time = time.perf_counter()
            outputs = model(inputs)
            
            if device.type == 'cuda': torch.cuda.synchronize()
            elif device.type == 'mps': torch.mps.synchronize()
            
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return {
        'y_true': np.array(all_labels),
        'y_pred': np.array(all_preds),
        'y_probs': np.array(all_probs),
        'latency_ms': np.mean(latencies) * 1000,
        'latency_std': np.std(latencies) * 1000
    }

def sanitize_config(d):
    for k, v in d.items():
        if isinstance(v, dict):
            sanitize_config(v)
        elif isinstance(v, str):
            if v.isdigit() or (v.startswith('-') and v[1:].isdigit()):
                d[k] = int(v)
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    pass
    return d

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERRO] Manifesto não encontrado.")
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest['weights_path'] != "MISSING"]
    evaluation_results = []

    print(f"\n>>> INICIANDO FASE 2: TRIBUNAL DE TESTE ({len(manifest)} modelos)")

    for idx, row in manifest.iterrows():
        m_type, version, weights_path, config_path = row['model_type'], row['dataset_version'], row['weights_path'], row['config_path']
        print(f"\n--- {m_type.upper()} ({version}) ---")

        try:
            config_obj = Config.load_model_config(m_type)
            model_config = config_obj.model_config
            
            with open(config_path, 'r') as f:
                opt_dict = yaml.safe_load(f)
            opt_dict = sanitize_config(opt_dict)
            
            if 'params' in opt_dict:
                for k, v in opt_dict['params'].items():
                    model_config.params[k] = v
            
            model_config.global_config['dataset_name'] = version
            input_channels = 15 if any(v in version for v in ["8", "11", "12"]) else 3
            test_loader, class_names = get_test_loader(version, input_channels)

            keys_to_clean = ['device', 'num_classes', 'input_channels']
            for k in keys_to_clean:
                if k in model_config.params: del model_config.params[k]

            model, _, _, _, _ = ModelFactory.create_pipeline(DEVICE, model_config, len(class_names), input_channels)
            
            ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt.state_dict()
            
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            
            # Cálculo de Complexidade (GFLOPs e Params)
            gflops, params_m = calcular_complexidade_modelo(model, num_channels=input_channels, device=DEVICE)
            
            results = run_benchmarking(model, test_loader, DEVICE)

            pred_file = f"{m_type}_{version}_preds.npz"
            np.savez_compressed(os.path.join(PREDICTIONS_DIR, pred_file), 
                               y_true=results['y_true'], 
                               y_pred=results['y_pred'], 
                               y_probs=results['y_probs'], 
                               class_names=class_names)

            evaluation_results.append({
                'model_type': m_type, 
                'dataset_version': version,
                'accuracy_test': (results['y_true'] == results['y_pred']).mean(),
                'latency_avg_ms': results['latency_ms'], 
                'params_m': params_m,
                'gflops': gflops,
                'prediction_file': pred_file
            })
            print(f"  [OK] Acc: {evaluation_results[-1]['accuracy_test']:.4f} | GFLOPs: {gflops:.4f}")

        except Exception as e:
            print(f"  [ERRO] {e}")

    if evaluation_results:
        pd.DataFrame(evaluation_results).to_csv(SUMMARY_PATH, index=False)
        print(f"\n>>> FASE 2 CONCLUÍDA.")

if __name__ == "__main__":
    main()
