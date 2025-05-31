import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.transforms import v2
from mlflow.models.signature import infer_signature
from glob import glob
from typing import Callable, Optional, Tuple, List
from torch.utils.data._utils.collate import default_collate

from src.config.models.config import Config

class NPYFolderDataset(Dataset):
    """
    Dataset PyTorch que carrega arquivos .npy organizados em subpastas por classe.
    Cada subpasta deve ter nome igual ao da classe e conter arquivos .npy.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.samples = []
        self.class_to_idx = {}
        self.transform = transform

        # Lista classes ordenadas e cria mapeamento
        classes = sorted([d for d in os.listdir(root_dir) 
                         if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # Carrega amostras
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            npy_files = glob(os.path.join(cls_dir, '*.npy'))
            for npy_path in npy_files:
                self.samples.append((npy_path, self.class_to_idx[cls_name]))
        
        self.classes = classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        npy_path, label = self.samples[idx]
        arr = np.load(npy_path)
        
        # Garante formato correto (C, H, W)
        if arr.ndim == 2:  # Grayscale
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[0] not in [1, 3]:  # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))  
        
        tensor = torch.from_numpy(arr).float()
        
        if self.transform:
            tensor = self.transform(tensor)
            
        return tensor, label

class TransformSubset(Dataset):
    """
    Permite aplicar transformações dinamicamente em subsets do Dataset.
    """
    def __init__(self, subset, transform: Optional[Callable] = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def compute_class_weights(targets: List[int], num_classes: int) -> torch.Tensor:
    """
    Calcula pesos de classe usando contagem eficiente com torch.bincount.
    """
    class_counts = torch.bincount(torch.tensor(targets, dtype=torch.long))
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    return class_weights

def load_datasets(batch_size: int, config, model_config) -> Tuple[DataLoader, DataLoader, int, torch.Tensor, List[str], np.ndarray]:
    """
    Prepara DataLoaders para treino e validação com amostragem balanceada.
    
    Retorna:
        Tuple[DataLoader, DataLoader, int, torch.Tensor, List[str]]: 
        (train_loader, val_loader, num_classes, class_weights, class_names)
    """

    # Carrega dataset completo
    full_dataset = NPYFolderDataset(root_dir=config.data_dir)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    temp_loader = DataLoader(
        full_dataset,
        batch_size=Config.batch_size,
        num_workers=config.num_workers,
        shuffle=False
    )

    norm_mean, norm_std = compute_stats(temp_loader, full_dataset)

    # Calcula pesos de classe
    targets = [label for _, label in full_dataset.samples]
    class_weights = compute_class_weights(targets, num_classes)
    sample_weights = class_weights[torch.tensor(targets, dtype=torch.long)]

    # Divide em treino/validação
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # Aplica transformações específicas
    if config.train_transform:
        train_dataset = TransformSubset(train_subset, transform=config.train_transform)
    else:
        train_dataset = TransformSubset(
            train_subset,
            v2.Compose(
                [v2.ToDtype(torch.float32), v2.Normalize(mean=norm_mean, std=norm_std)]
            ),
        )

    if config.val_transform:
        val_dataset = TransformSubset(val_subset, transform=config.val_transform)
    else:
        val_dataset = TransformSubset(
            val_subset,
            v2.Compose(
                [v2.ToDtype(torch.float32), v2.Normalize(mean=norm_mean, std=norm_std)]
            ),
        )

    # Configura amostrador balanceado
    train_sampler = WeightedRandomSampler(
        weights=sample_weights[train_subset.indices],
        num_samples=len(train_subset),
        replacement=True
    )

    # Augmentation
    image_transforms, batch_transforms = get_augmentations(model_config, num_classes)

    # Aplicar image-level transforms no dataset
    if image_transforms:
        train_transform = v2.Compose(image_transforms)
        train_dataset = TransformSubset(train_subset, transform=train_transform)

    # Cria transformação combinada de batch-level com random choice
    if batch_transforms:
        cutmix_or_mixup = v2.RandomChoice(batch_transforms)
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))
    else:
        collate_fn = None


    # Cria DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=collate_fn  
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )

    sample_input, _ = full_dataset[0]
    input_example = sample_input.unsqueeze(0).to(Config.device).cpu().numpy().astype(np.float32)
    input_schema = infer_signature(input_example)

    return train_loader, val_loader, num_classes, class_weights, class_names, input_schema


def compute_stats(loader: DataLoader, full_dataset) -> Tuple[list, list]:
    mean = torch.zeros(full_dataset[0][0].shape[0])  # Pega número de canais automaticamente
    std = torch.zeros_like(mean)
    total_pixels = 0

    for batch, _ in loader:
        batch = batch.view(-1, batch.size(1), *batch.shape[2:])  # (N, C, H, W)
        n_pixels = batch.size(0) * batch.size(2) * batch.size(3)
        
        # Acumula somas
        mean += batch.mean(dim=(0, 2, 3)) * n_pixels
        std += batch.std(dim=(0, 2, 3)) * n_pixels
        total_pixels += n_pixels

    mean /= total_pixels
    std /= total_pixels

    return mean.tolist(), std.tolist()


def get_augmentations(model_config, num_classes) -> Tuple[List, List]:

    image_transforms = []
    batch_transforms = []

    if model_config.augmentations:
        for augmentation in model_config.augmentations:
            aug_type = augmentation['type']
            aug_params = augmentation.get('params', {})

            # === BATCH-LEVEL AUGMENTATIONS ===
            if aug_type in ["CutMix", "MixUp"]:
                aug_params_with_classes = {**aug_params, 'num_classes': num_classes}
                prob = aug_params_with_classes.pop('prob', 1.0)
                transform_class = getattr(v2, aug_type)
                transform_instance = transform_class(**aug_params_with_classes)
                batch_transforms.append(
                    v2.RandomApply([transform_instance], p=prob)
                )

            # === IMAGE-LEVEL AUGMENTATIONS ===
            else:
                try:
                    transform_class = getattr(v2, aug_type)
                    transform_instance = transform_class(**aug_params)
                    image_transforms.append(transform_instance)
                    
                except AttributeError:
                    print(f"Warning: Transformação '{aug_type}' não encontrada em torchvision.transforms.v2")
                    continue

    return image_transforms, batch_transforms