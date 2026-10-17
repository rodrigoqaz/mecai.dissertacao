"""
Função objetivo para otimização de hiperparâmetros com Optuna.
"""

import optuna
import yaml
import traceback
from pathlib import Path
from typing import Dict, Any
from src.optimization.hyperparameter_config import OptimizationConfig
from src.optimization.dynamic_config import create_dynamic_config
from src.utils.mlflow_logger import MLflowLogger

from src.optimization.hyperparameter_config import OptimizationConfig, OptimizationObjectives

class OptimizationObjective:
    """
    Classe que implementa a função objetivo para otimização com Optuna.
    """
    
    def __init__(self, model_name: str, base_config, objective_type: str = "accuracy", study_name: str = None):
        """
        Inicializa a função objetivo.
        
        Args:
            model_name: Nome do modelo (resnet, inception, etc.)
            base_config: Configuração base carregada do YAML
            objective_type: Tipo de objetivo (accuracy, balanced, loss)
            study_name: Nome do estudo Optuna (usado como nome do experimento MLflow)
        """
        self.model_name = model_name
        self.base_config = base_config
        self.objective_type = objective_type
        self.study_name = study_name # Armazena o nome do estudo
    
    def __call__(self, trial: optuna.Trial) -> float:
        """
        Função objetivo chamada pelo Optuna para cada trial.
        
        Args:
            trial: Trial do Optuna
            
        Returns:
            Valor do objetivo a ser maximizado
        """
        try:
            # Gera hiperparâmetros para este trial
            optimization_params = OptimizationConfig.get_search_space(
                self.model_name, trial
            )
            
            # Cria configuração dinâmica
            dynamic_config = create_dynamic_config(
                self.model_name, optimization_params
            )
            
            # Nome do experimento para este trial
            # Usa o study_name passado para a classe
            dynamic_config.experiment_name = self.study_name
            
            # Executa treinamento
            metrics = self._train_model(dynamic_config, trial.number)
            
            # Calcula objetivo baseado no tipo especificado
            objective_value = self._calculate_objective(
                metrics['val_accuracy'], 
                metrics['val_loss'],
                all_preds=metrics.get('val_preds'),  
                all_labels=metrics.get('val_labels')
            )
            
            return objective_value
            
        except Exception as e:
            print(f"Erro no trial {trial.number}: {str(e)}")
            traceback.print_exc()
            # Retorna valor baixo para trials que falham
            return 0.0
    
    def _train_model(self, config, trial_number: int) -> Dict[str, float]:
        """
        Executa o treinamento do modelo e retorna métricas.
        
        Args:
            config: Configuração dinâmica
            trial_number: Número do trial
            
        Returns:
            Dicionário com métricas finais
        """
        # Import aqui para evitar dependências circulares
        from train_model_v2 import train_single_model
        
        print(f"\n--- TRIAL {trial_number:03d} ---")
        print(f"Modelo: {self.model_name}")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.model_config.optimizer['params']['lr']:.6f}")
        
        # Adiciona tags para identificar trials de otimização
        tags = {
            "optuna_trial": "true",
            "trial_number": str(trial_number),
            "model_type": self.model_name,
            "objective_type": self.objective_type
        }
        
        # Temporariamente modifica o método start_run para incluir tags
        original_experiment_name = config.experiment_name
        config.experiment_name = original_experiment_name
        
        # Executa treinamento
        metrics = train_single_model(config)
        
        print(f"Trial {trial_number:03d} | "
            f"Acc: {metrics['val_accuracy']:.2f}% | "
            f"Loss: {metrics['val_loss']:.4f} | "
            f"Ep: {metrics['best_epoch']}")
        
        return metrics
    
    def _calculate_objective(self, val_acc: float, val_loss: float, all_preds=None, all_labels=None) -> float:
        """
        Calcula o objetivo delegando para a classe científica OptimizationObjectives.
        """
        if self.objective_type == "accuracy":
            return OptimizationObjectives.accuracy_objective(val_acc, val_loss)
            
        elif self.objective_type == "loss":
            return OptimizationObjectives.loss_objective(val_acc, val_loss)
            
        elif self.objective_type == "mcc":
            return OptimizationObjectives.mcc_objective(
                val_acc, val_loss, all_preds=all_preds, all_labels=all_labels
            )
            
        elif self.objective_type == "mcc_loss_composite": 
            return OptimizationObjectives.mcc_loss_composite(
                val_acc, val_loss, all_preds=all_preds, all_labels=all_labels
            )
            
        elif self.objective_type == "balanced":
             return OptimizationObjectives.balanced_harmonic_mean(val_acc, val_loss)
             
        else:
            return val_acc
    
    def save_best_config(self, best_params: Dict[str, Any], filename: str):
        """
        Salva a melhor configuração encontrada em arquivo YAML.
        
        Args:
            best_params: Melhores parâmetros encontrados
            filename: Nome do arquivo para salvar
        """
        try:
            # Cria configuração dinâmica com os melhores parâmetros
            optimization_params = self._convert_params_to_config_format(best_params)
            dynamic_config = create_dynamic_config(self.model_name, optimization_params)
            
            # Estrutura a configuração no formato YAML
            config_dict = {
                "model_type": self.model_name,
                "global_config": {
                    "data_dir": self.base_config.data_dir,
                    "experiment_name": self.study_name, # Usa o study_name aqui
                    "mlflow_uri": self.base_config.mlflow_uri,
                    "dataset_name": getattr(self.base_config, 'dataset_name', 'v5'),
                    "max_epochs": self.base_config.max_epochs,
                    "batch_size": dynamic_config.batch_size,
                    "num_workers": getattr(self.base_config, 'num_workers', 0),
                    "device": "auto"
                },
                "params": dynamic_config.model_config.params,
                "optimizer": dynamic_config.model_config.optimizer,
                "scheduler": dynamic_config.model_config.scheduler,
                "loss": dynamic_config.model_config.loss,

                "augmentation_config": getattr(dynamic_config, 'augmentation_config', {}), # Add augmentation config
                "callbacks": getattr(self.base_config.model_config, 'callbacks', [])
            }
            
            # Salva arquivo
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                
            print(f"Configuração otimizada salva em: {filename}")
            
        except Exception as e:
            print(f"Erro ao salvar configuração: {str(e)}")
    
    def _convert_params_to_config_format(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converte parâmetros do Optuna para formato de configuração.
        """
        config_format = {}
        model_prefix = self.model_name
        
        # Batch size
        batch_size_key = f"{model_prefix}_batch_size"
        if batch_size_key in params:
            config_format["batch_size"] = params[batch_size_key]
        
        # Parâmetros do modelo
        model_params = {}
        param_mappings = {
            f"{model_prefix}_architecture": "architecture",
            f"{model_prefix}_unfreeze_layers": "unfreeze_layers", 
            f"{model_prefix}_hidden_units": "hidden_units",
            f"{model_prefix}_dropout": "dropout"
        }
        
        for optuna_key, config_key in param_mappings.items():
            if optuna_key in params:
                model_params[config_key] = params[optuna_key]
        
        if model_params:
            config_format["params"] = {
                **self.base_config.model_config.params,
                **model_params
            }
        
        # Otimizador
        optimizer_type_key = f"{model_prefix}_optimizer_type"
        lr_key = f"{model_prefix}_lr"
        weight_decay_key = f"{model_prefix}_weight_decay"
        
        if optimizer_type_key in params or lr_key in params or weight_decay_key in params:
            config_format["optimizer"] = {
                "type": params.get(optimizer_type_key, self.base_config.model_config.optimizer["type"]),
                "params": {
                    "lr": params.get(lr_key, self.base_config.model_config.optimizer["params"]["lr"]),
                    "weight_decay": params.get(weight_decay_key, self.base_config.model_config.optimizer["params"]["weight_decay"])
                }
            }
        
        # Scheduler
        scheduler_type_key = f"{model_prefix}_scheduler_type"
        if scheduler_type_key in params:
            config_format["scheduler"] = {
                "type": params[scheduler_type_key],
                "params": self._build_scheduler_params(params, model_prefix)
            }
        
        # Loss
        label_smoothing_key = f"{model_prefix}_label_smoothing"
        if label_smoothing_key in params:
            config_format["loss"] = {
                "type": "CrossEntropyLoss",
                "params": {
                    "label_smoothing": params[label_smoothing_key]
                }
            }
            
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation_key = f"{model_prefix}_use_augmentation"
        if use_augmentation_key in params:
            augmentation_config["use_augmentation"] = params[use_augmentation_key]
            
            if augmentation_config["use_augmentation"]:
                per_image_aug_method_key = f"{model_prefix}_per_image_aug_method"
                if per_image_aug_method_key in params:
                    augmentation_config["per_image_aug_method"] = params[per_image_aug_method_key]
                    if params[per_image_aug_method_key] == "basic":
                        augmentation_config["basic_params"] = {
                            "n_augments": params.get(f"{model_prefix}_n_augments"),
                            "rotation_range": params.get(f"{model_prefix}_rotation_range"),
                            "horizontal_flip_prob": params.get(f"{model_prefix}_horizontal_flip_prob"),
                            "brightness_range": params.get(f"{model_prefix}_brightness_range"),
                            "contrast_range": params.get(f"{model_prefix}_contrast_range"),
                            "width_shift_range": params.get(f"{model_prefix}_width_shift_range")
                        }
                    elif params[per_image_aug_method_key] == "advanced":
                        augmentation_config["advanced_params"] = {} # Advanced method currently has no tunable params
                        
                batch_level_aug_method_key = f"{model_prefix}_batch_level_aug_method"
                if batch_level_aug_method_key in params:
                    augmentation_config["batch_level_aug_method"] = params[batch_level_aug_method_key]
                    if params[batch_level_aug_method_key] in ["cutmix", "mixup"]:
                        augmentation_config["mix_alpha"] = params.get(f"{model_prefix}_mix_alpha")

        if augmentation_config:
            config_format["augmentation_config"] = augmentation_config
        
        return config_format
    
    def _build_scheduler_params(self, params: Dict[str, Any], model_prefix: str) -> Dict[str, Any]:
        """
        Constrói parâmetros do scheduler baseado no tipo.
        """
        scheduler_type_key = f"{model_prefix}_scheduler_type"
        scheduler_type = params.get(scheduler_type_key, self.base_config.model_config.scheduler["type"])
        
        if scheduler_type == "CosineAnnealingWarmRestarts":
            return {
                "T_0": params.get(f"{model_prefix}_T_0", 10),
                "T_mult": params.get(f"{model_prefix}_T_mult", 2),
                "eta_min": 1e-6
            }
        elif scheduler_type == "ReduceLROnPlateau":
            return {
                "mode": "min",
                "patience": params.get(f"{model_prefix}_scheduler_patience", 5),
                "factor": params.get(f"{model_prefix}_scheduler_factor", 0.5),
                "min_lr": 1e-6
            }
        elif scheduler_type == "StepLR":
            return {
                "step_size": params.get(f"{model_prefix}_step_size", 20),
                "gamma": params.get(f"{model_prefix}_gamma", 0.3)
            }
        else:
            return self.base_config.model_config.scheduler.get("params", {})
    

