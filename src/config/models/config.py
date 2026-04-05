import yaml
import torch
from pathlib import Path
from torchvision import transforms
from typing import Dict, Any
from torchvision.transforms import v2

class TransformFactory:
    """Factory para criar transformações do torchvision a partir de configurações YAML"""
    
    @staticmethod
    def create_transform(transform_config: Dict[str, Any]) -> transforms.Compose:
        """
        Cria uma transformação composta a partir de configuração YAML.
        
        Args:
            transform_config: Dicionário com lista de transformações
            
        Returns:
            transforms.Compose com as transformações especificadas
        """
        if not transform_config or not transform_config.get('transforms'):
            return None
            
        transform_list = []
        
        for transform_def in transform_config['transforms']:
            transform_name = transform_def['name']
            transform_params = transform_def.get('params', {})
            
            # Tenta buscar primeiro em torchvision.transforms
            try:
                transform_class = getattr(transforms, transform_name)
            except AttributeError:
                # Se não encontrar, tenta em torchvision.transforms.v2
                try:
                    transform_class = getattr(v2, transform_name)
                except AttributeError:
                    print(f"Warning: Transformação '{transform_name}' não encontrada")
                    continue
            
            # Cria a instância da transformação
            transform_instance = transform_class(**transform_params)
            transform_list.append(transform_instance)
        
        return transforms.Compose(transform_list) if transform_list else None



class ModelConfig:
    """Classe para armazenar configurações de modelo"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        # Configurações globais
        self.global_config = config_dict.get('global_config', {})
        
        # Configurações específicas do modelo
        self.model_type = config_dict.get('model_type')
        self.params = config_dict.get('params', {})
        self.optimizer = config_dict.get('optimizer', {})
        self.scheduler = config_dict.get('scheduler', {})
        self.loss = config_dict.get('loss', {})
        self.augmentations = config_dict.get('augmentations', {})
        self.callbacks = config_dict.get('callbacks', {})
        
        # Transformações
        self.train_transform = TransformFactory.create_transform(
            config_dict.get('train_transform')
        )
        self.val_transform = TransformFactory.create_transform(
            config_dict.get('val_transform')
        )
        
        # Atributos de configuração global como propriedades da classe
        for key, value in self.global_config.items():
            setattr(self, key, value)
        
        if 'input_channels' in config_dict:
            setattr(self, 'input_channels', config_dict['input_channels'])


class Config:
    """
    Classe de configuração genérica que carrega tudo do YAML.
    """
    
    def __init__(self, model_config: ModelConfig):
        """Inicializa com configurações do modelo"""
        self.model_config = model_config
        
        # Aplica configurações globais como atributos da classe
        for key, value in model_config.global_config.items():
            setattr(self, key, value)
        
        if hasattr(model_config, 'input_channels'):
            setattr(self, 'input_channels', model_config.input_channels)
        
        # Configurações de hardware/device
        self.device = self._get_device()
        
        # Transformações
        self.train_transform = model_config.train_transform
        self.val_transform = model_config.val_transform
    
    def _get_device(self) -> torch.device:
        """Determina o device apropriado"""
        device_config = self.model_config.global_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_config)
    
    @staticmethod
    def load_model_config(model_name: str) -> 'Config':
        """
        Carrega a configuração do modelo e a configuração da versão do dado,
        unificando-as em um único objeto de configuração.
        
        Args:
            model_name: Nome do modelo (arquivo YAML sem extensão)
            
        Returns:
            Instância de Config com todas as configurações carregadas
        """
        
        # 1. Carrega a configuração principal do modelo
        model_config_path = Path(f"src/config/models/models/{model_name}.yaml")
        
        if not model_config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {model_config_path}")
        
        with open(model_config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 2. Verifica se há uma configuração de dataset associada
        dataset_name = config_data.get('global_config', {}).get('dataset_name')

        if dataset_name:
             data_version_path = Path(f"src/config/data/versions/{dataset_name}.yaml")
             if data_version_path.exists():
                 with open(data_version_path, 'r', encoding='utf-8') as f:
                     data_version_config = yaml.safe_load(f)
 
                 # 3. Unifica as configurações.
                 # Começa com a config do dado e atualiza com a do modelo.
                 # Isso garante que configurações no arquivo do modelo (mais específico)
                 # tenham prioridade em caso de chaves duplicadas.
                 merged_config = data_version_config.copy()
                 merged_config.update(config_data)
                 config_data = merged_config
             else:
                 print(f"AVISO: Arquivo de configuração para o dataset '{dataset_name}' não encontrado em {data_version_path}.")
        
        model_config = ModelConfig(config_data)
        return Config(model_config)
    
    def get_device(self) -> torch.device:
        """Retorna o device configurado"""
        return self.device