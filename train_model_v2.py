import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score
)
import mlflow
from sklearn.calibration import CalibrationDisplay

from src.config.models.config import Config
from src.data.data_loader import load_datasets
from src.utils.trainer_v2 import train_epoch, validate
from models.model_factory import ModelFactory
from src.utils.mlflow_logger import MLflowLogger
from typing import Dict, Union
from src.optimization.dynamic_config import DynamicConfig


def train_single_model(config: Union[Config, DynamicConfig]) -> Dict[str, float]:
    """
    Função de treinamento que pode ser usada tanto para treinamento
    tradicional quanto para otimização de hiperparâmetros.
    """
    
    tags = None
    if isinstance(config, DynamicConfig):
        tags = {
            "optuna_optimization": "true",
            "model_type": config.base_config.model_config.model_type if hasattr(config.base_config.model_config, 'model_type') else "unknown"
        }
    
    MLflowLogger.set_tracking_uri(config.mlflow_uri)
    seed = config.model_config.global_config.get('seed', 42)
    generator = torch.Generator().manual_seed(seed)
    
    if mlflow.active_run():
        print("Aviso: Fechando run anterior pendente.")
        mlflow.end_run()
    
    MLflowLogger.start_run(config.experiment_name, tags=tags)

    try: 

        train_loader, val_loader, num_classes, input_channels, class_weights, class_names, input_schema = (
            load_datasets(config.batch_size, config, generator=generator)
        )

        model, optimizer, scheduler, criterion, callbacks = ModelFactory.create_pipeline(
            config.device, config.model_config, num_classes, input_channels
        )

        log_params = {
            'batch_size': config.batch_size,
            'num_classes': num_classes,
            'max_epochs': config.max_epochs,
            'device': str(config.device),
            'seed': seed
        }
        if hasattr(config.model_config, '__dict__'):
            for key, value in config.model_config.__dict__.items():
                if not key.startswith('_') and not callable(value) and isinstance(value, (str, int, float, bool, list, dict)):
                    log_params[key] = value
        
        # Log augmentation parameters if available
        if hasattr(config, 'augmentation_config') and config.augmentation_config:
            log_params['augmentation_config'] = config.augmentation_config

        MLflowLogger.log_params(log_params)

        best_val_acc = 0.0
        best_val_acc_epoch = 0
        best_val_loss = float('inf')
        final_train_loss = 0.0
        final_train_acc = 0.0
        
        best_epoch_labels = None
        best_epoch_predictions = None
        best_epoch_probabilities = None

        for epoch in range(config.max_epochs):
            train_loss, train_acc = train_epoch(epoch, config.max_epochs, model, train_loader, criterion, optimizer, config.device)
            final_train_loss = train_loss
            final_train_acc = train_acc
            
            val_loss, val_report, epoch_labels, epoch_preds, epoch_probs = validate(model, val_loader, criterion, class_names, config.device)

            MLflowLogger.log_metrics({
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_report['accuracy'] * 100,
                'train/lr': optimizer.param_groups[0]['lr']
            }, step=epoch)

            for cls in class_names:
                MLflowLogger.log_metrics({
                    f'class/precision_{cls}': val_report[cls]['precision'],
                    f'class/recall_{cls}': val_report[cls]['recall'],
                    f'class/f1_{cls}': val_report[cls]['f1-score']
                }, step=epoch)

            if val_report['accuracy'] * 100 > best_val_acc:
                best_val_acc = val_report['accuracy'] * 100
                best_val_acc_epoch = epoch
                best_val_loss = val_loss

                MLflowLogger.log_model(model, "best_model", input_schema)
                
                best_epoch_labels = epoch_labels
                best_epoch_predictions = epoch_preds
                best_epoch_probabilities = epoch_probs

            for cb in callbacks:
                cb(val_loss, model)

            if any(getattr(cb, "early_stop", False) for cb in callbacks):
                print("Early stopping acionado.")
                for cb in callbacks:
                    if hasattr(cb, "restore_best_weights"): cb.restore_best_weights(model)
                break

            scheduler.step(val_loss)

        # --- Bloco final de avaliação e logging ---
        if best_epoch_labels is not None:
            print(f"\n--- Avaliação Final (Melhor Época: {best_val_acc_epoch}) ---")
            
            # Obter caminho do run para salvar artefatos
            run = MLflowLogger.get_active_run()
            run_id = run.info.run_id
            artifact_path = mlflow.get_artifact_uri().replace("file://", "")
            os.makedirs(artifact_path, exist_ok=True)

            # 1. Relatório de Classificação e Métricas Agregadas
            final_report = classification_report(best_epoch_labels, best_epoch_predictions, target_names=class_names, output_dict=True, zero_division=0)
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in final_report:
                    MLflowLogger.log_metrics({
                        f'final/{avg_type}_precision': final_report[avg_type]['precision'],
                        f'final/{avg_type}_recall': final_report[avg_type]['recall'],
                        f'final/{avg_type}_f1-score': final_report[avg_type]['f1-score'],
                    }, step=best_val_acc_epoch)

            # 2. Matriz de Confusão
            cm = confusion_matrix(best_epoch_labels, best_epoch_predictions)
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_path = os.path.join(artifact_path, "confusion_matrix.csv")
            cm_df.to_csv(cm_path)
            MLflowLogger.log_artifact(cm_path, "final_artifacts")
            print("Matriz de confusão registrada.")

            # 3. Métricas Robustas (Kappa e MCC)
            kappa = cohen_kappa_score(best_epoch_labels, best_epoch_predictions)
            mcc = matthews_corrcoef(best_epoch_labels, best_epoch_predictions)
            MLflowLogger.log_metrics({'final/cohen_kappa': kappa, 'final/matthews_corrcoef': mcc}, step=best_val_acc_epoch)
            print(f"Cohen's Kappa: {kappa:.4f} | Matthews CorrCoef: {mcc:.4f}")

            # 4. ROC AUC (se houver probabilidades)
            if best_epoch_probabilities is not None:
                if num_classes > 2:
                    macro_roc_auc = roc_auc_score(best_epoch_labels, best_epoch_probabilities, multi_class='ovr', average='macro')
                else: # Caso binário
                    macro_roc_auc = roc_auc_score(best_epoch_labels, np.array(best_epoch_probabilities)[:, 1])
                MLflowLogger.log_metrics({'final/macro_roc_auc': macro_roc_auc}, step=best_val_acc_epoch)
                print(f"Macro ROC AUC: {macro_roc_auc:.4f}")

                # 5. Gráfico de Calibração
                plt.figure(figsize=(10, 7))
                ax = plt.gca()
                plt.title("Diagrama de Confiabilidade")

                for i in range(num_classes):
                    # Binariza os rótulos para a classe atual
                    y_true_binary = (np.array(best_epoch_labels) == i).astype(int)
                    # Pega as probabilidades para a classe atual
                    y_prob_class = np.array(best_epoch_probabilities)[:, i]
                    
                    # Plota a curva de calibração para a classe
                    display = CalibrationDisplay.from_predictions(
                        y_true_binary,
                        y_prob_class,
                        n_bins=10,
                        ax=ax,
                        name=class_names[i] 
                    )
                
                calib_plot_path = os.path.join(artifact_path, "calibration_plot.png")
                plt.savefig(calib_plot_path)
                plt.close()
                MLflowLogger.log_artifact(calib_plot_path, "final_artifacts")
                print("Diagrama de Confiabilidade registrado.")
            
            print("--- Fim da Avaliação ---")

        final_metrics = {
            'val/final_accuracy': best_val_acc,
            'val/best_epoch': best_val_acc_epoch
        }
        MLflowLogger.log_metrics(final_metrics, step=best_val_acc_epoch)
        MLflowLogger.log_model(model, "final_model", input_schema)
        MLflowLogger.end_run()

        print(f"\nTreinamento finalizado. Melhor acurácia na época {best_val_acc_epoch}: {best_val_acc:.2f}%")
        
        return {
            'val_accuracy': best_val_acc,
            'val_loss': best_val_loss,
            'val_preds': best_epoch_predictions, 
            'val_labels': best_epoch_labels,
            'train_accuracy': final_train_acc,
            'train_loss': final_train_loss,
            'best_epoch': best_val_acc_epoch
        }
    
    except Exception as e:
        print(f"Erro crítico durante o treinamento: {e}")
        raise e 

    finally:
        if mlflow.active_run():
            MLflowLogger.end_run()

def main(model_name: str = "resnet"):
    config = Config.load_model_config(model_name)
    train_single_model(config)


if __name__ == "__main__":

    # models = ['densenet', 'efficientnet', 'inception', 'resnet', 'vgg', 'convnext''swin', ]
    models = ['efficientnet']
    for model in models:
        main(model)
    # main("vgg")
