from src.config.models.config import Config
# from src.config.data.config import ConfigurationManager
from src.data.data_loader import load_datasets
from src.utils.trainer import train_epoch, validate
from models.model_factory import ModelFactory
from src.utils.mlflow_logger import MLflowLogger
from typing import Dict, Union
from src.optimization.dynamic_config import DynamicConfig
import torch


def train_single_model(config: Union[Config, DynamicConfig]) -> Dict[str, float]:
    """
    Função de treinamento que pode ser usada tanto para treinamento
    tradicional quanto para otimização de hiperparâmetros.
    
    Args:
        config: Configuração do modelo (Config ou DynamicConfig)
        
    Returns:
        Dict com métricas finais: {'val_accuracy', 'val_loss', 'train_accuracy', 'train_loss'}
    """
    
    # Define tags para o MLflow se for um trial de otimização
    tags = None
    if isinstance(config, DynamicConfig):
        tags = {
            "optuna_optimization": "true",
            "model_type": config.base_config.model_config.model_type if hasattr(config.base_config.model_config, 'model_type') else "unknown"
        }
    
    # Define o tracking URI do MLflow
    MLflowLogger.set_tracking_uri(config.mlflow_uri)

    seed = config.model_config.global_config.get('seed', 42) # Pega a seed ou usa 42 como padrão
    generator = torch.Generator().manual_seed(seed)
    
    # Inicia run no MLflow
    MLflowLogger.start_run(config.experiment_name, tags=tags)

    # Carrega dados
    train_loader, val_loader, num_classes, input_channels, class_weights, class_names, input_schema = (
        load_datasets(config.batch_size, config, config.model_config, generator=generator)
    )

    # Modelo, otimizador, scheduler
    # model, optimizer, scheduler, criterion, callbacks = ModelFactory.create_pipeline(
    #     config.device, config.model_config, num_classes
    # )
    model, optimizer, scheduler, criterion, callbacks = ModelFactory.create_pipeline(
        config.device, config.model_config, num_classes, input_channels
    )

    # Log de hiperparâmetros
    log_params = {
        'batch_size': config.batch_size,
        'num_classes': num_classes,
        'max_epochs': config.max_epochs,
        'device': str(config.device),
        'seed': seed
    }
    
    # Adiciona parâmetros do modelo_config
    if hasattr(config.model_config, '__dict__'):
        for key, value in config.model_config.__dict__.items():
            if not key.startswith('_') and not callable(value):
                try:
                    # Serializa apenas tipos básicos
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        log_params[key] = value
                except:
                    pass  # Ignora valores que não podem ser serializados
    
    MLflowLogger.log_params(log_params)

    best_val_acc = 0.0
    best_val_acc_epoch = 0
    final_train_loss = 0.0
    final_train_acc = 0.0

    # Loop de treinamento
    for epoch in range(config.max_epochs):
        train_loss, train_acc = train_epoch(
            epoch,
            config.max_epochs,
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
        )
        val_loss, val_acc, val_report = validate(
            model, 
            val_loader, 
            criterion, 
            class_names, 
            config.device
        )

        # Atualiza métricas finais
        final_train_loss = train_loss
        final_train_acc = train_acc

        # Logging de métricas globais
        MLflowLogger.log_metrics({
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'val/loss': val_loss,
            'val/accuracy': val_acc,
            'train/lr': optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Logging de métricas por classe
        for cls in class_names:
            MLflowLogger.log_metrics({
                f'class/precision_{cls}': val_report[cls]['precision'],
                f'class/recall_{cls}': val_report[cls]['recall'],
                f'class/f1_{cls}': val_report[cls]['f1-score']
            }, step=epoch)

        # Log do melhor modelo no MLflow
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            MLflowLogger.log_model(model, "best_model", input_schema)

        # Executa callbacks
        for cb in callbacks:
            cb(val_loss, model)

        # Verifica early stopping
        early_stop = False
        for cb in callbacks:
            if hasattr(cb, "early_stop") and getattr(cb, "early_stop"):
                print("Early stopping acionado.")
                if hasattr(cb, "restore_best_weights"):
                    cb.restore_best_weights(model)
                early_stop = True
                break

        if early_stop:
            break

        # Atualiza scheduler
        scheduler.step(val_loss)

    # Log final
    final_metrics = {
        'val/final_accuracy': best_val_acc,
        'val/final_loss': val_loss,
        'val/best_epoch': best_val_acc_epoch
    }
    
    MLflowLogger.log_metrics(final_metrics, step=best_val_acc_epoch)
    MLflowLogger.log_model(model, "final_model", input_schema)
    MLflowLogger.end_run()

    print(f"Treinamento finalizado. Melhor acurácia: {best_val_acc:.4f}")
    
    # Retorna métricas para otimização
    return {
        'val_accuracy': best_val_acc,
        'val_loss': val_loss,
        'train_accuracy': final_train_acc,
        'train_loss': final_train_loss,
        'best_epoch': best_val_acc_epoch
    }


def main(model_name: str = "resnet"):
    """
    Função principal de treinamento usando configuração 100% baseada em YAML.
    Mantida para compatibilidade com execução tradicional.
    
    Args:
        model_name: Nome do modelo (corresponde ao arquivo YAML)
    """
    config = Config.load_model_config(model_name)
    train_single_model(config)


if __name__ == "__main__":
    main("resnet")
