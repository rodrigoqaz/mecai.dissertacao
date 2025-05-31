from src.config.models.config import Config
from src.data.data_loader import load_datasets
from src.utils.trainer import train_epoch, validate
from models.model_factory import ModelFactory
from src.utils.mlflow_logger import MLflowLogger

def main(model_name: str = "vgg"):

    model_config = Config.load_model_config(model_name)
    
    # Inicia run no MLflow 0K
    MLflowLogger.start_run(Config.experiment_name)

    # Carrega dados OK
    train_loader, val_loader, num_classes, class_weights, class_names, input_schema = load_datasets(Config.batch_size, Config, model_config)

    # Modelo, otimizador, scheduler OK
    model, optimizer, scheduler, criterion, callbacks = ModelFactory.create_pipeline(model_config, num_classes)

    # Log de hiperparâmetros
    MLflowLogger.log_params({
        **model_config.__dict__,
        'batch_size': Config.batch_size,
        'num_classes': num_classes,
        'max_epochs': Config.max_epochs
    })

    best_val_acc = 0.0

    for epoch in range(Config.max_epochs):
        train_loss, train_acc = train_epoch(epoch, Config.max_epochs, model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_report = validate(model, val_loader, criterion, class_names)

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

        # Log do melhor modelo no MLflow (opcional: só se for o melhor val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            MLflowLogger.log_model(model, "best_model", input_schema)

        # Callbacks
        for cb in callbacks:
            cb(val_loss, model)

        # Checa se algum callback ativou early stopping
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

        scheduler.step(val_loss)


    if not early_stop:  # Só loga se completou todas as épocas
        MLflowLogger.log_metrics({'val/final_accuracy': best_val_acc}, step=best_val_acc_epoch)

    # MLflowLogger.log_metrics({'val/final_accuracy': best_val_acc}, step=best_val_acc_epoch)
    MLflowLogger.log_model(model, "final_model", input_schema)
    MLflowLogger.end_run()

    print("Treinamento finalizado.")

if __name__ == "__main__":
    main("densenet")
