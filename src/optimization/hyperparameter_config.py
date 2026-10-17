"""
Configurações para otimização de hiperparâmetros com Optuna.
Define os espaços de busca para cada tipo de modelo.
"""

from typing import Dict, Any, List, Optional
import optuna
from sklearn.metrics import f1_score, matthews_corrcoef
import numpy as np


# class OptimizationConfig:
#     """
#     Define espaços de busca para otimização de hiperparâmetros por modelo.
#     """
    
#     @staticmethod
#     def get_search_space(model_type: str, trial: optuna.Trial) -> Dict[str, Any]:
#         """
#         Retorna o espaço de busca para um tipo de modelo específico.
        
#         Args:
#             model_type: Tipo do modelo (resnet, inception, etc.)
#             trial: Trial do Optuna para sugerir valores
            
#         Returns:
#             Dicionário com hiperparâmetros sugeridos
#         """
#         if model_type == "resnet":
#             return OptimizationConfig._get_resnet_search_space(trial)
#         elif model_type == "inception":
#             return OptimizationConfig._get_inception_search_space(trial)
#         elif model_type == "efficientnet":
#             return OptimizationConfig._get_efficientnet_search_space(trial)
#         elif model_type == "densenet":
#             return OptimizationConfig._get_densenet_search_space(trial)
#         elif model_type == "vgg":
#             return OptimizationConfig._get_vgg_search_space(trial)
#         elif model_type == "vit":
#             return OptimizationConfig._get_vit_search_space(trial)
#         elif model_type == "swin":
#             return OptimizationConfig._get_swin_search_space(trial)
#         elif model_type == "convnext":
#             return OptimizationConfig._get_convnext_search_space(trial)
#         else:
#             raise ValueError(f"Modelo '{model_type}' não suportado para otimização")
    
#     @staticmethod
#     def _get_vit_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para Vision Transformer (ViT)"""
#         return {
#             "batch_size": trial.suggest_categorical("vit_batch_size", [16, 32, 48]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("vit_architecture", ["vit_b_16", "vit_b_32"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("vit_unfreeze_layers", 2, 6),
#                 "hidden_units": trial.suggest_categorical("vit_hidden_units", [256, 512, 768]),
#                 "dropout": trial.suggest_float("vit_dropout", 0.3, 0.7),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("vit_optimizer_type", ["AdamW"]), # AdamW é comum para transformers
#                 "params": {
#                     "lr": trial.suggest_float("vit_lr", 1e-5, 1e-3, log=True),
#                     "weight_decay": trial.suggest_float("vit_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": trial.suggest_categorical("vit_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "vit")
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("vit_label_smoothing", 0.0, 0.2)
#                 }
#             }
#         }

#     @staticmethod
#     def _get_swin_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para Swin Transformer"""
#         return {
#             "batch_size": trial.suggest_categorical("swin_batch_size", [16, 32, 48]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("swin_architecture", ["swin_t", "swin_s"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("swin_unfreeze_layers", 2, 6),
#                 "hidden_units": trial.suggest_categorical("swin_hidden_units", [256, 512, 768]),
#                 "dropout": trial.suggest_float("swin_dropout", 0.3, 0.7),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("swin_optimizer_type", ["AdamW"]), # AdamW é comum para transformers
#                 "params": {
#                     "lr": trial.suggest_float("swin_lr", 1e-5, 1e-3, log=True),
#                     "weight_decay": trial.suggest_float("swin_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": trial.suggest_categorical("swin_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "swin")
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("swin_label_smoothing", 0.0, 0.2)
#                 }
#             }
#         }

#     @staticmethod
#     def _get_convnext_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para ConvNeXt"""
#         return {
#             "batch_size": trial.suggest_categorical("convnext_batch_size", [16, 32, 64]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("convnext_architecture", ["convnext_tiny", "convnext_small", "convnext_base"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("convnext_unfreeze_layers", 3, 9), # Mais camadas para descongelar
#                 "hidden_units": trial.suggest_categorical("convnext_hidden_units", [512, 1024, 2048]),
#                 "dropout": trial.suggest_float("convnext_dropout", 0.3, 0.7),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("convnext_optimizer_type", ["Adam", "AdamW", "SGD"]),
#                 "params": {
#                     "lr": trial.suggest_float("convnext_lr", 1e-5, 1e-2, log=True),
#                     "weight_decay": trial.suggest_float("convnext_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": trial.suggest_categorical("convnext_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "StepLR"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "convnext")
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("convnext_label_smoothing", 0.0, 0.3)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_resnet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para ResNet"""
#         return {
#             # Configurações globais
#             "batch_size": trial.suggest_categorical("resnet_batch_size", [16, 32, 64]),
            
#             # Parâmetros do modelo
#             "params": {
#                 "architecture": trial.suggest_categorical("resnet_architecture", ["resnet18", "resnet34", "resnet50", "resnet101"]),
#                 "pretrained": True,  # Manter fixo
#                 "unfreeze_layers": trial.suggest_int("resnet_unfreeze_layers", 2, 12),
#                 "hidden_units": trial.suggest_categorical("resnet_hidden_units", [256, 512, 1024, 2048]),
#                 "dropout": trial.suggest_float("resnet_dropout", 0.3, 0.8),
#                 "weights": "IMAGENET1K_V1"  # Manter fixo
#             },
            
#             # Otimizador
#             "optimizer": {
#                 "type": trial.suggest_categorical("resnet_optimizer_type", ["Adam", "AdamW", "SGD"]),
#                 "params": {
#                     "lr": trial.suggest_float("resnet_lr", 1e-5, 1e-2, log=True),
#                     "weight_decay": trial.suggest_float("resnet_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             # Scheduler
#             "scheduler": {
#                 "type": trial.suggest_categorical("resnet_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "StepLR"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "resnet")
#             },
            
#             # Loss
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("resnet_label_smoothing", 0.0, 0.3)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_inception_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para Inception"""
#         return {
#             # Configurações globais
#             "batch_size": trial.suggest_categorical("inception_batch_size", [16, 32, 64]),
            
#             # Parâmetros do modelo
#             "params": {
#                 "architecture": "inception_v3",  # Fixo para Inception
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("inception_unfreeze_layers", 4, 16),
#                 "hidden_units": trial.suggest_categorical("inception_hidden_units", [512, 1024, 2048]),
#                 "dropout": trial.suggest_float("inception_dropout", 0.4, 0.8),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             # Otimizador
#             "optimizer": {
#                 "type": trial.suggest_categorical("inception_optimizer_type", ["Adam", "AdamW"]),
#                 "params": {
#                     "lr": trial.suggest_float("inception_lr", 1e-5, 5e-3, log=True),
#                     "weight_decay": trial.suggest_float("inception_weight_decay", 1e-6, 5e-3, log=True)
#                 }
#             },
            
#             # Scheduler
#             "scheduler": {
#                 "type": "ReduceLROnPlateau",  # Fixo para Inception
#                 "params": {
#                     "mode": "min",
#                     "patience": trial.suggest_int("inception_scheduler_patience", 3, 10),
#                     "factor": trial.suggest_float("inception_scheduler_factor", 0.3, 0.7),
#                     "min_lr": 1e-6
#                 }
#             },
            
#             # Loss
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("inception_label_smoothing", 0.05, 0.25)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_efficientnet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para EfficientNet"""
#         return {
#             "batch_size": trial.suggest_categorical("efficientnet_batch_size", [16, 32, 48]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("efficientnet_architecture", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("efficientnet_unfreeze_layers", 3, 10),
#                 "hidden_units": trial.suggest_categorical("efficientnet_hidden_units", [256, 512, 1024]),
#                 "dropout": trial.suggest_float("efficientnet_dropout", 0.3, 0.7),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("efficientnet_optimizer_type", ["Adam", "AdamW"]),
#                 "params": {
#                     "lr": trial.suggest_float("efficientnet_lr", 1e-5, 1e-2, log=True),
#                     "weight_decay": trial.suggest_float("efficientnet_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": trial.suggest_categorical("efficientnet_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "efficientnet")
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("efficientnet_label_smoothing", 0.0, 0.2)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_densenet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para DenseNet"""
#         return {
#             "batch_size": trial.suggest_categorical("densenet_batch_size", [16, 32, 64]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("densenet_architecture", ["densenet121", "densenet161", "densenet169"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("densenet_unfreeze_layers", 2, 8),
#                 "hidden_units": trial.suggest_categorical("densenet_hidden_units", [512, 1024, 2048]),
#                 "dropout": trial.suggest_float("densenet_dropout", 0.3, 0.7),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("densenet_optimizer_type", ["Adam", "AdamW", "SGD"]),
#                 "params": {
#                     "lr": trial.suggest_float("densenet_lr", 1e-5, 1e-2, log=True),
#                     "weight_decay": trial.suggest_float("densenet_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": trial.suggest_categorical("densenet_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]),
#                 "params": OptimizationConfig._get_scheduler_params(trial, "densenet")
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("densenet_label_smoothing", 0.0, 0.3)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_vgg_search_space(trial: optuna.Trial) -> Dict[str, Any]:
#         """Espaço de busca para VGG"""
#         return {
#             "batch_size": trial.suggest_categorical("vgg_batch_size", [16, 32, 48]),
            
#             "params": {
#                 "architecture": trial.suggest_categorical("vgg_architecture", ["vgg11", "vgg13", "vgg16", "vgg19"]),
#                 "pretrained": True,
#                 "unfreeze_layers": trial.suggest_int("vgg_unfreeze_layers", 2, 6),
#                 "hidden_units": trial.suggest_categorical("vgg_hidden_units", [512, 1024, 2048, 4096]),
#                 "dropout": trial.suggest_float("vgg_dropout", 0.4, 0.8),
#                 "weights": "IMAGENET1K_V1"
#             },
            
#             "optimizer": {
#                 "type": trial.suggest_categorical("vgg_optimizer_type", ["Adam", "AdamW", "SGD"]),
#                 "params": {
#                     "lr": trial.suggest_float("vgg_lr", 1e-5, 1e-2, log=True),
#                     "weight_decay": trial.suggest_float("vgg_weight_decay", 1e-6, 1e-2, log=True)
#                 }
#             },
            
#             "scheduler": {
#                 "type": "ReduceLROnPlateau",  # Fixo para VGG
#                 "params": {
#                     "mode": "min",
#                     "patience": trial.suggest_int("vgg_scheduler_patience", 5, 15),
#                     "factor": trial.suggest_float("vgg_scheduler_factor", 0.1, 0.5),
#                     "min_lr": 1e-7
#                 }
#             },
            
#             "loss": {
#                 "type": "CrossEntropyLoss",
#                 "params": {
#                     "label_smoothing": trial.suggest_float("vgg_label_smoothing", 0.0, 0.2)
#                 }
#             }
#         }
    
#     @staticmethod
#     def _get_scheduler_params(trial: optuna.Trial, model_prefix: str) -> Dict[str, Any]:
#         """Retorna parâmetros específicos para cada tipo de scheduler"""
#         # Obtém o scheduler_type que foi sugerido anteriormente no mesmo trial
#         scheduler_type = None
#         for param_name, param_value in trial.params.items():
#             if param_name == f"{model_prefix}_scheduler_type":
#                 scheduler_type = param_value
#                 break
        
#         # Se ainda não encontrou, usa fallback
#         if scheduler_type is None:
#             scheduler_type = "ReduceLROnPlateau"
        
#         if scheduler_type == "CosineAnnealingWarmRestarts":
#             return {
#                 "T_0": trial.suggest_int(f"{model_prefix}_T_0", 5, 20),
#                 "T_mult": trial.suggest_int(f"{model_prefix}_T_mult", 1, 3),
#                 "eta_min": 1e-6
#             }
#         elif scheduler_type == "ReduceLROnPlateau":
#             return {
#                 "mode": "min",
#                 "patience": trial.suggest_int(f"{model_prefix}_scheduler_patience", 3, 10),
#                 "factor": trial.suggest_float(f"{model_prefix}_scheduler_factor", 0.2, 0.7),
#                 "min_lr": 1e-6
#             }
#         elif scheduler_type == "StepLR":
#             return {
#                 "step_size": trial.suggest_int(f"{model_prefix}_step_size", 10, 30),
#                 "gamma": trial.suggest_float(f"{model_prefix}_gamma", 0.1, 0.5)
#             }
#         else:
#             return {}

from typing import Dict, Any
import optuna

class OptimizationConfig:
    """
    Define espaços de busca cientificamente ajustados para otimização de hiperparâmetros.
    Considera input de 15 canais e fine-tuning.
    """

    @staticmethod
    def _get_advanced_aug_params(trial: optuna.Trial, prefix: str) -> Dict[str, Any]:
        """
        Define o espaço de busca para augmentação avançada.
        Controla a probabilidade e intensidade das distorções.
        """
        return {
            # Probabilidade de aplicar distorções geométricas (Grid, Elastic, Optical)
            "p_distort": trial.suggest_float(f"{prefix}_aug_p_distort", 0.2, 0.5),
            
            # Probabilidade de aplicar ruídos (Gauss, Multiplicative)
            "p_noise": trial.suggest_float(f"{prefix}_aug_p_noise", 0.1, 0.4),
            
            # Intensidade do CoarseDropout (buracos na imagem)
            "p_dropout": trial.suggest_float(f"{prefix}_aug_p_dropout", 0.1, 0.3),
            "num_holes": trial.suggest_int(f"{prefix}_aug_num_holes", 4, 12),
            
            # Limites do RandomResizedCrop (Zoom)
            "crop_scale_min": trial.suggest_float(f"{prefix}_aug_crop_scale_min", 0.6, 0.8)
        }
    
    @staticmethod
    def get_search_space(model_type: str, trial: optuna.Trial) -> Dict[str, Any]:
        if model_type == "resnet":
            return OptimizationConfig._get_resnet_search_space(trial)
        elif model_type == "inception":
            return OptimizationConfig._get_inception_search_space(trial)
        elif model_type == "efficientnet":
            return OptimizationConfig._get_efficientnet_search_space(trial)
        elif model_type == "densenet":
            return OptimizationConfig._get_densenet_search_space(trial)
        elif model_type == "vgg":
            return OptimizationConfig._get_vgg_search_space(trial)
        elif model_type == "vit":
            return OptimizationConfig._get_vit_search_space(trial)
        elif model_type == "swin":
            return OptimizationConfig._get_swin_search_space(trial)
        elif model_type == "convnext":
            return OptimizationConfig._get_convnext_search_space(trial)
        else:
            raise ValueError(f"Modelo '{model_type}' não suportado")

    # =========================================================================
    # GRUPO A: TRANSFORMERS (Sensíveis, AdamW, Regularização Alta)
    # =========================================================================
    
    @staticmethod
    def _get_vit_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        # Sugestão de LR baseada no scheduler para evitar erro lógico
        sched_type = trial.suggest_categorical("vit_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"])
        
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("vit_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("vit_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("vit_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("vit_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("vit_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("vit_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("vit_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("vit_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "vit")
                
            batch_level_aug_method = trial.suggest_categorical("vit_batch_level_aug_method", ["cutmix", "mixup"])#, "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("vit_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("vit_batch_size", [16, 32]), # 15 canais pesam na VRAM
            "params": {
                "architecture": trial.suggest_categorical("vit_architecture", ["vit_b_16", "vit_b_32"]),
                "pretrained": True,
                # Transformers aprendem rápido. Descongelar muito cedo pode degradar features pré-treinadas.
                "unfreeze_layers": trial.suggest_int("vit_unfreeze_layers", 1, 4), 
                "hidden_units": trial.suggest_categorical("vit_hidden_units", [256, 512]),
                "dropout": trial.suggest_float("vit_dropout", 0.0, 0.3), # Transformers usam menos dropout no head, mais weight decay
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": "AdamW", # SGD quase nunca converge bem em ViT
                "params": {
                    "lr": trial.suggest_float("vit_lr", 1e-5, 2e-4, log=True), # LR mais baixo que CNNs
                    "weight_decay": trial.suggest_float("vit_weight_decay", 0.01, 0.1, log=True) # Alta regularização
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "vit", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": trial.suggest_float("vit_label_smoothing", 0.0, 0.1)}
            },
            "augmentation_config": augmentation_config
        }

    @staticmethod
    def _get_swin_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        sched_type = trial.suggest_categorical("swin_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"])
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("swin_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("swin_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("swin_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("swin_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("swin_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("swin_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("swin_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("swin_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "swin")
                
            batch_level_aug_method = trial.suggest_categorical("swin_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("swin_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("swin_batch_size", [16, 32]),
            "params": {
                "architecture": trial.suggest_categorical("swin_architecture", ["swin_t", "swin_s"]),
                "pretrained": True,
                "unfreeze_layers": trial.suggest_int("swin_unfreeze_layers", 1, 4),
                "hidden_units": trial.suggest_categorical("swin_hidden_units", [256, 512]),
                "dropout": trial.suggest_float("swin_dropout", 0.0, 0.3),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": trial.suggest_float("swin_lr", 1e-5, 2e-4, log=True),
                    "weight_decay": trial.suggest_float("swin_weight_decay", 0.01, 0.1, log=True)
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "swin", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": trial.suggest_float("swin_label_smoothing", 0.0, 0.1)}
            },
            "augmentation_config": augmentation_config
        }

    # =========================================================================
    # GRUPO B: CNNs MODERNAS (ConvNeXt, EfficientNet)
    # =========================================================================

    @staticmethod
    def _get_convnext_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        sched_type = trial.suggest_categorical("convnext_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"])
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("convnext_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("convnext_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("convnext_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("convnext_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("convnext_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("convnext_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("convnext_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("convnext_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "convnext")
                
            batch_level_aug_method = trial.suggest_categorical("convnext_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("convnext_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("convnext_batch_size", [16, 32]),
            "params": {
                "architecture": trial.suggest_categorical("convnext_architecture", ["convnext_tiny", "convnext_small"]),
                "pretrained": True,
                "unfreeze_layers": trial.suggest_int("convnext_unfreeze_layers", 2, 6),
                "hidden_units": trial.suggest_categorical("convnext_hidden_units", [512, 1024]),
                "dropout": trial.suggest_float("convnext_dropout", 0.2, 0.5),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": "AdamW", # ConvNeXt foi desenhada para AdamW
                "params": {
                    "lr": trial.suggest_float("convnext_lr", 1e-4, 4e-3, log=True),
                    "weight_decay": trial.suggest_float("convnext_weight_decay", 1e-3, 0.05, log=True)
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "convnext", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": trial.suggest_float("convnext_label_smoothing", 0.1, 0.2)}
            },
            "augmentation_config": augmentation_config
        }

    @staticmethod
    def _get_efficientnet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        sched_type = trial.suggest_categorical("efficientnet_scheduler_type", ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"])
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("efficientnet_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("efficientnet_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("efficientnet_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("efficientnet_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("efficientnet_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("efficientnet_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("efficientnet_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("efficientnet_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "efficientnet")
                
            batch_level_aug_method = trial.suggest_categorical("efficientnet_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("efficientnet_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("efficientnet_batch_size", [16, 32, 48]),
            "params": {
                "architecture": trial.suggest_categorical("efficientnet_architecture", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]),
                "pretrained": True,
                "unfreeze_layers": trial.suggest_int("efficientnet_unfreeze_layers", 2, 8),
                "hidden_units": trial.suggest_categorical("efficientnet_hidden_units", [256, 512]),
                "dropout": trial.suggest_float("efficientnet_dropout", 0.2, 0.5),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": trial.suggest_categorical("efficientnet_optimizer_type", ["AdamW", "Adam"]),
                "params": {
                    "lr": trial.suggest_float("efficientnet_lr", 1e-4, 1e-2, log=True),
                    "weight_decay": trial.suggest_float("efficientnet_weight_decay", 1e-5, 1e-3, log=True)
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "efficientnet", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.1}
            },
            "augmentation_config": augmentation_config
        }

    # =========================================================================
    # GRUPO C: CNNs ROBUSTAS (ResNet, DenseNet, Inception)
    # =========================================================================

    @staticmethod
    def _get_resnet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        sched_type = trial.suggest_categorical("resnet_scheduler_type", ["ReduceLROnPlateau", "StepLR"])
        # ResNet funciona muito bem com SGD + Momentum
        opt_type = trial.suggest_categorical("resnet_optimizer_type", ["AdamW", "SGD"])
        
        lr_low = 1e-4 if opt_type == "AdamW" else 1e-3
        lr_high = 1e-3 if opt_type == "AdamW" else 1e-1
        
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("resnet_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("resnet_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("resnet_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("resnet_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("resnet_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("resnet_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("resnet_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("resnet_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "resnet")
                
            batch_level_aug_method = trial.suggest_categorical("resnet_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("resnet_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("resnet_batch_size", [32, 64]),
            "params": {
                "architecture": trial.suggest_categorical("resnet_architecture", ["resnet18", "resnet34", "resnet50"]),
                "pretrained": True,
                # ResNet tem 4 stages. Descongelar > 3 stages é quase treinar do zero.
                "unfreeze_layers": trial.suggest_int("resnet_unfreeze_layers", 1, 3), 
                "hidden_units": trial.suggest_categorical("resnet_hidden_units", [256, 512]),
                "dropout": trial.suggest_float("resnet_dropout", 0.3, 0.6),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": opt_type,
                "params": {
                    "lr": trial.suggest_float("resnet_lr", lr_low, lr_high, log=True),
                    "weight_decay": trial.suggest_float("resnet_weight_decay", 1e-5, 1e-3, log=True),
                    # Se for SGD, adicionamos momentum fixo pois é padrão
                    **({"momentum": 0.9} if opt_type == "SGD" else {})
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "resnet", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.1}
            },
            "augmentation_config": augmentation_config
        }

    @staticmethod
    def _get_inception_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        # # Inception é conhecida por gostar de decaimento de LR suave (RMSprop ou Step)
        # sched_type = "ReduceLROnPlateau"
        # return {
        #     "batch_size": trial.suggest_categorical("inception_batch_size", [32, 64]),
        #     "params": {
        #         "architecture": "inception_v3",
        #         "pretrained": True,
        #         # Inception é profunda e larga.
        #         "unfreeze_layers": trial.suggest_int("inception_unfreeze_layers", 2, 6),
        #         "hidden_units": trial.suggest_categorical("inception_hidden_units", [512, 1024]),
        #         "dropout": trial.suggest_float("inception_dropout", 0.3, 0.6),
        #         "weights": "IMAGENET1K_V1"
        #     },
        #     "optimizer": {
        #         "type": "AdamW",
        #         "params": {
        #             "lr": trial.suggest_float("inception_lr", 1e-4, 1e-3, log=True),
        #             "weight_decay": trial.suggest_float("inception_weight_decay", 1e-5, 1e-3, log=True)
        #         }
        #     },
        #     "scheduler": {
        #         "type": sched_type,
        #         "params": OptimizationConfig._get_scheduler_params(trial, "inception", sched_type)
        #     },
        #     "loss": {
        #         "type": "CrossEntropyLoss",
        #         "params": {"label_smoothing": 0.1}
        #     }
        # }

        sched_type = trial.suggest_categorical("inception_scheduler_type", ["ReduceLROnPlateau"])
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("inception_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("inception_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("inception_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("inception_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("inception_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("inception_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("inception_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("inception_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "densenet")
                
            batch_level_aug_method = trial.suggest_categorical("densenet_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("densenet_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("inception_batch_size", [32, 64]),
            "params": {
                "architecture": "inception_v3",
                "pretrained": True,
                # Inception é profunda e larga.
                "unfreeze_layers": trial.suggest_int("inception_unfreeze_layers", 2, 6),
                "hidden_units": trial.suggest_categorical("inception_hidden_units", [512, 1024]),
                "dropout": trial.suggest_float("inception_dropout", 0.3, 0.6),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": trial.suggest_float("inception_lr", 1e-4, 1e-3, log=True),
                    "weight_decay": trial.suggest_float("inception_weight_decay", 1e-5, 1e-3, log=True)
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "inception", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.1}
            },
            "augmentation_config": augmentation_config
        }

    @staticmethod
    def _get_densenet_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        sched_type = trial.suggest_categorical("densenet_scheduler_type", ["ReduceLROnPlateau"])
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("densenet_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("densenet_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("densenet_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("densenet_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("densenet_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("densenet_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("densenet_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("densenet_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "densenet")
                
            batch_level_aug_method = trial.suggest_categorical("densenet_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("densenet_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("densenet_batch_size", [16, 32]), # Heavy on memory
            "params": {
                "architecture": trial.suggest_categorical("densenet_architecture", ["densenet121", "densenet169"]),
                "pretrained": True,
                "unfreeze_layers": trial.suggest_int("densenet_unfreeze_layers", 2, 6),
                "hidden_units": trial.suggest_categorical("densenet_hidden_units", [256, 512]),
                "dropout": trial.suggest_float("densenet_dropout", 0.2, 0.5),
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": trial.suggest_float("densenet_lr", 1e-4, 1e-2, log=True),
                    "weight_decay": trial.suggest_float("densenet_weight_decay", 1e-5, 1e-3, log=True)
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "densenet", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.1}
            },
            "augmentation_config": augmentation_config
        }

    # =========================================================================
    # GRUPO D: LEGACY (VGG)
    # =========================================================================

    @staticmethod
    def _get_vgg_search_space(trial: optuna.Trial) -> Dict[str, Any]:
        # VGG é lenta e pesada.
        sched_type = "ReduceLROnPlateau"
        
        # Parâmetros de Augmentação
        augmentation_config = {}
        use_augmentation = trial.suggest_categorical("vgg_use_augmentation", [True, False])
        augmentation_config["use_augmentation"] = use_augmentation
        
        if use_augmentation:
            per_image_aug_method = trial.suggest_categorical("vgg_per_image_aug_method", ["basic", "advanced", "none"])
            augmentation_config["per_image_aug_method"] = per_image_aug_method
            
            if per_image_aug_method == "basic":
                augmentation_config["basic_params"] = {
                    "n_augments": trial.suggest_int("vgg_n_augments", 1, 3),
                    "rotation_range": trial.suggest_int("vgg_rotation_range", 0, 45),
                    "horizontal_flip_prob": trial.suggest_float("vgg_horizontal_flip_prob", 0.0, 1.0),
                    "brightness_range": trial.suggest_float("vgg_brightness_range", 0.0, 0.3),
                    "contrast_range": trial.suggest_float("vgg_contrast_range", 0.0, 0.3),
                    "width_shift_range": trial.suggest_float("vgg_width_shift_range", 0.0, 0.2)
                }
            elif per_image_aug_method == "advanced":
                augmentation_config["advanced_params"] = OptimizationConfig._get_advanced_aug_params(trial, "vgg")
                
            batch_level_aug_method = trial.suggest_categorical("vgg_batch_level_aug_method", ["cutmix", "mixup", "none"])
            augmentation_config["batch_level_aug_method"] = batch_level_aug_method
            
            if batch_level_aug_method in ["cutmix", "mixup"]:
                augmentation_config["mix_alpha"] = trial.suggest_float("vgg_mix_alpha", 0.2, 1.0)
        
        return {
            "batch_size": trial.suggest_categorical("vgg_batch_size", [16, 32]),
            "params": {
                "architecture": trial.suggest_categorical("vgg_architecture", ["vgg16_bn", "vgg19_bn"]), # Use Batch Norm versions!
                "pretrained": True,
                "unfreeze_layers": trial.suggest_int("vgg_unfreeze_layers", 2, 6),
                # VGG tem muito parametro no head. Reduza hidden_units para evitar overfitting.
                "hidden_units": trial.suggest_categorical("vgg_hidden_units", [256, 512]), 
                "dropout": trial.suggest_float("vgg_dropout", 0.4, 0.7), # Precisa de dropout alto
                "weights": "IMAGENET1K_V1"
            },
            "optimizer": {
                "type": trial.suggest_categorical("vgg_optimizer_type", ["SGD", "AdamW"]),
                "params": {
                    "lr": trial.suggest_float("vgg_lr", 1e-4, 1e-2, log=True),
                    "weight_decay": trial.suggest_float("vgg_weight_decay", 1e-4, 1e-2, log=True),
                    **({"momentum": 0.9} if "SGD" in ["SGD"] else {}) # Simplificacao logica
                }
            },
            "scheduler": {
                "type": sched_type,
                "params": OptimizationConfig._get_scheduler_params(trial, "vgg", sched_type)
            },
            "loss": {
                "type": "CrossEntropyLoss",
                "params": {"label_smoothing": 0.0} # VGG às vezes prefere hard labels
            },
            "augmentation_config": augmentation_config
        }

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    @staticmethod
    def _get_scheduler_params(trial: optuna.Trial, model_prefix: str, scheduler_type: str) -> Dict[str, Any]:
        """
        Retorna parâmetros baseados no TIPO já escolhido anteriormente.
        Evita o erro de tentar ler parâmetros que ainda não foram amostrados.
        """
        if scheduler_type == "CosineAnnealingWarmRestarts":
            return {
                "T_0": trial.suggest_int(f"{model_prefix}_T_0", 5, 15),
                "T_mult": 1, # Simplificação para fine-tuning curto
                "eta_min": 1e-6
            }
        elif scheduler_type == "ReduceLROnPlateau":
            return {
                "mode": "min",
                "patience": trial.suggest_int(f"{model_prefix}_scheduler_patience", 3, 8),
                "factor": trial.suggest_float(f"{model_prefix}_scheduler_factor", 0.1, 0.5),
                "min_lr": 1e-7
            }
        elif scheduler_type == "StepLR":
            return {
                "step_size": trial.suggest_int(f"{model_prefix}_step_size", 5, 15),
                "gamma": 0.1
            }
        return {}


# class OptimizationObjectives:
#     """
#     Define diferentes objetivos de otimização.
#     """
    
#     @staticmethod
#     def accuracy_objective(val_acc: float, val_loss: float) -> float:
#         """Otimiza apenas accuracy de validação (maximizar)"""
#         return val_acc
    
#     @staticmethod
#     def balanced_objective(val_acc: float, val_loss: float, alpha: float = 0.7) -> float:
#         """Otimiza combinação balanceada de accuracy e loss"""
#         # Normalizar loss para estar na mesma escala que accuracy
#         normalized_loss = 1.0 / (1.0 + val_loss)
#         return alpha * val_acc + (1 - alpha) * normalized_loss
    
#     @staticmethod
#     def loss_objective(val_acc: float, val_loss: float) -> float:
#         """Otimiza apenas loss de validação (minimizar - Optuna inverte automaticamente)"""
#         return -val_loss  # Negativo porque Optuna maximiza por padrão


class OptimizationObjectives:
    
    @staticmethod
    def accuracy_objective(val_acc: float, val_loss: float, **kwargs) -> float:
        return val_acc
    
    @staticmethod
    def loss_objective(val_acc: float, val_loss: float, **kwargs) -> float:
        # Optuna maximiza por padrão, então retornamos negativo
        return -val_loss

    @staticmethod
    def f1_macro_objective(val_acc: float, val_loss: float, all_preds=None, all_labels=None) -> float:
        """
        Otimiza F1-Score Macro. 
        Crucial se suas classes (floresta, água, cidade, etc) não tiverem a mesma qtde de imagens.
        """
        if all_preds is None or all_labels is None:
            return val_acc # Fallback se não tiver predições brutas
            
        return f1_score(all_labels, all_preds, average='macro')

    @staticmethod
    def balanced_harmonic_mean(val_acc: float, val_loss: float, **kwargs) -> float:
        """
        Substitui sua soma linear por algo estatisticamente mais robusto.
        Tenta maximizar Acurácia enquanto penaliza Loss alta.
        """
        # Convertemos Loss para uma escala de "qualidade" [0, 1] usando exponencial negativa
        # Se loss = 0 -> quality = 1. Se loss -> infinito, quality -> 0
        loss_quality = np.exp(-val_loss)
        
        # Média harmônica (como no F1-score) pune valores baixos em qualquer um dos lados
        # Se a Acurácia for alta mas a Loss for alta (modelo inseguro), o score cai drasticamente.
        return 2 * (val_acc * loss_quality) / (val_acc + loss_quality + 1e-8)
    
    @staticmethod
    def mcc_objective(val_acc: float, val_loss: float, 
                      all_preds: Optional[List] = None, 
                      all_labels: Optional[List] = None, **kwargs) -> float:
        """
        Otimiza o Matthews Correlation Coefficient.
        Ideal para classes desbalanceadas. Retorna entre -1 e +1.
        """
        # Proteção: Se não tiver os vetores, faz fallback para acurácia
        if all_preds is None or all_labels is None:
            return val_acc
            
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return mcc - (0.1 * val_loss)
    
    @staticmethod
    def mcc_loss_composite(val_acc: float, val_loss: float, 
                           all_preds: Optional[List] = None, 
                           all_labels: Optional[List] = None, **kwargs) -> float:
        """
        Otimiza MCC mas usa a Loss para desempatar e suavizar a busca.
        Fórmula: 70% MCC + 30% Qualidade da Loss
        """
        if all_preds is None or all_labels is None:
            mcc = val_acc # Fallback
        else:
            mcc = matthews_corrcoef(all_labels, all_preds)
            
        # Normaliza MCC de [-1, 1] para [0, 1] para somar com a loss
        mcc_norm = (mcc + 1) / 2
        
        # Converte loss para escala de qualidade [0, 1] (menor loss = maior qualidade)
        loss_quality = np.exp(-val_loss)
        
        # Média ponderada (dá peso maior para a métrica de classificação)
        return 0.7 * mcc_norm + 0.3 * loss_quality