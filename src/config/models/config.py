import yaml
import torch
from pathlib import Path
from torchvision import transforms

class ModelConfig:
    def __init__(self, model_type: str, params: dict, optimizer: dict, scheduler: dict, loss: dict, augmentations:dict, callbacks: dict):
        self.model_type = model_type
        self.params = params
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.augmentations = augmentations
        self.callbacks = callbacks


class Config:
    """
    Configurações globais e hiperparâmetros do pipeline de treinamento.
    """

    # Diretório dos dados (subpastas por classe, arquivos .npy)
    data_dir = 'data/gold/datasets/v5/'

    # Nome do experimento e diretório do MLflow
    experiment_name = "DenseNet GPU - Aumento de Dados(v1)"
    mlflow_uri = "file:///Users/rodrigoqaz/Documents/tmp/mlruns"

    # Nome do dataset
    dataset_name = "v5"

    # Treinamento
    max_epochs = 70
    batch_size = 32

    # Hardware
    num_workers = 0
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Augmentation (habilitar/desabilitar)
    aug = False

    # Transforms para treino e validação (compatíveis com VGG/ResNet)
    # train_transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    # val_transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    train_transform = None
    val_transform = None

    @staticmethod
    def get_device():
        return Config.device
    
    @staticmethod
    def load_model_config(model_name: str):
        config_path = Path(f"src/config/models/models/{model_name}.yaml")
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return ModelConfig(**config_data)
