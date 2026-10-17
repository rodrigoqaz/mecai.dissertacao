"""
Classe para criar configurações dinâmicas que podem ser modificadas pelo Optuna.
"""

import copy
from typing import Dict, Any
from src.config.models.config import ModelConfig, Config


class DynamicConfig:
    """
    Cria configurações dinâmicas que podem ser modificadas durante a otimização.
    Herda de Config para compatibilidade completa.
    """
    
    def __init__(self, base_config: Config, optimization_params: Dict[str, Any]):
        """
        Inicializa configuração dinâmica baseada numa configuração base.
        
        Args:
            base_config: Configuração base carregada do YAML
            optimization_params: Parâmetros sugeridos pelo Optuna
        """
        self.base_config = base_config
        self.optimization_params = optimization_params
        
        # Cria uma cópia profunda da configuração base
        self._create_dynamic_config()
    
    def _create_dynamic_config(self):
        """Cria configuração dinâmica aplicando parâmetros de otimização"""
        # Copia as configurações base
        self.model_config = copy.deepcopy(self.base_config.model_config)
        
        # Aplica parâmetros globais
        if "batch_size" in self.optimization_params:
            self.batch_size = self.optimization_params["batch_size"]
        else:
            self.batch_size = self.base_config.batch_size
        
        # Aplica parâmetros do modelo
        if "params" in self.optimization_params:
            for key, value in self.optimization_params["params"].items():
                setattr(self.model_config, key, value)
                # Também atualiza o dicionário params interno
                if hasattr(self.model_config, 'params') and isinstance(self.model_config.params, dict):
                    self.model_config.params[key] = value
        
        # Aplica configurações do otimizador
        if "optimizer" in self.optimization_params:
            self.model_config.optimizer = self.optimization_params["optimizer"]
        
        # Aplica configurações do scheduler
        if "scheduler" in self.optimization_params:
            self.model_config.scheduler = self.optimization_params["scheduler"]
        
        # Aplica configurações da loss
        if "loss" in self.optimization_params:
            self.model_config.loss = self.optimization_params["loss"]
        
        # Aplica outras configurações globais
        for attr in ['max_epochs', 'device', 'experiment_name', 'data_dir', 'num_workers', 'mlflow_uri']:
            if hasattr(self.base_config, attr):
                setattr(self, attr, getattr(self.base_config, attr))
        
        # Mantém transformações da configuração base
        # self.train_transform = self.base_config.train_transform # Handled dynamically
        # self.val_transform = self.base_config.val_transform     # Handled dynamically
        
        # Aplica configurações de augmentação
        if "augmentation_config" in self.optimization_params:
            self.augmentation_config = self.optimization_params["augmentation_config"]
        else:
            self.augmentation_config = {}
        
        # Mantém device da configuração base
        self.device = self.base_config.device
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Retorna um dicionário com todas as configurações para logging.
        """
        config_dict = {
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'device': str(self.device)
        }
        
        # Adiciona parâmetros do modelo
        if hasattr(self.model_config, 'params'):
            config_dict.update(self.model_config.params)
        
        # Adiciona parâmetros do otimizador
        if hasattr(self.model_config, 'optimizer'):
            config_dict['optimizer_type'] = self.model_config.optimizer.get('type', 'Unknown')
            config_dict.update({f"optimizer_{k}": v for k, v in self.model_config.optimizer.get('params', {}).items()})
        
        # Adiciona parâmetros do scheduler
        if hasattr(self.model_config, 'scheduler'):
            config_dict['scheduler_type'] = self.model_config.scheduler.get('type', 'Unknown')
            config_dict.update({f"scheduler_{k}": v for k, v in self.model_config.scheduler.get('params', {}).items()})
        
        # Adiciona parâmetros da loss
        if hasattr(self.model_config, 'loss'):
            config_dict['loss_type'] = self.model_config.loss.get('type', 'CrossEntropyLoss')
            config_dict.update({f"loss_{k}": v for k, v in self.model_config.loss.get('params', {}).items()})
        
        return config_dict
    
    def get_experiment_name(self, trial_number: int) -> str:
        """
        Gera nome do experimento incluindo número do trial.
        """
        base_name = getattr(self, 'experiment_name', 'OptimizationStudy')
        return f"{base_name}_Trial_{trial_number:03d}"

    def __getattr__(self, name):
        """
        Mágica: Se o atributo não existir aqui (ex: data_dir, seed),
        busca automaticamente no base_config.
        """
        return getattr(self.base_config, name)

    def __setattr__(self, name, value):
        """
        Permite definir atributos. Se for atributo interno, define aqui.
        Se for atributo do base_config, define lá.
        """
        internal_attrs = ['base_config', 'optimization_params', 'model_config', 'batch_size']
        
        if name in internal_attrs:
            super().__setattr__(name, value)
        else:
            # Se não for interno, tentamos definir no base_config também para manter sincronia
            # Mas cuidado: preferimos manter as alterações locais se possível.
            # Para simplificar e evitar recursão infinita no __init__:
            super().__setattr__(name, value)

def create_dynamic_config(model_name: str, optimization_params: Dict[str, Any]) -> DynamicConfig:
    """
    Factory function para criar configuração dinâmica.
    
    Args:
        model_name: Nome do modelo base
        optimization_params: Parâmetros de otimização do Optuna
        
    Returns:
        DynamicConfig configurada
    """
    # Carrega configuração base
    base_config = Config.load_model_config(model_name)
    
    # Cria configuração dinâmica
    return DynamicConfig(base_config, optimization_params)
