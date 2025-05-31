import torch.nn as nn
import torch.optim as optim
from importlib import import_module


class ModelFactory:
    @staticmethod
    def instantiate_callbacks(callbacks_config):
        callbacks = []
        for cb in callbacks_config:
            module_path, class_name = cb['class_path'].rsplit('.', 1)
            module = import_module(module_path)
            callback_cls = getattr(module, class_name)
            init_args = cb.get('init_args', {})
            callbacks.append(callback_cls(**init_args))
        return callbacks

    @staticmethod
    def create_pipeline(model_config, num_classes):
        # Modelo
        module = import_module(f"models.{model_config.model_type}")
        model = module.initialize_model(num_classes, **model_config.params)
        # Otimizador
        opt_class = getattr(optim, model_config.optimizer['type'])
        optimizer = opt_class(model.parameters(), **model_config.optimizer['params'])
        # Scheduler
        sch_class = getattr(optim.lr_scheduler, model_config.scheduler['type'])
        scheduler = sch_class(optimizer, **model_config.scheduler['params'])
        # Loss
        loss_name = model_config.loss.get('type', 'CrossEntropyLoss')
        loss_params = model_config.loss.get('params', {})
        criterion = getattr(nn, loss_name)(**loss_params)
        # Callbacks
        callbacks = []
        if hasattr(model_config, "callbacks"):
            callbacks = ModelFactory.instantiate_callbacks(model_config.callbacks)
        return model, optimizer, scheduler, criterion, callbacks
