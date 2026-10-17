import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision.transforms import v2
import albumentations as A
from mlflow.models.signature import infer_signature
from glob import glob
from typing import Callable, Optional, Tuple, List, Any, Dict
from torch.utils.data._utils.collate import default_collate
from src.config.models.config import Config
from src.data.processors.augmentations import AugmentationManager, CutMixWrapper, MixUpWrapper
from torchvision import tv_tensors

class AlbumentationsWrapper(torch.nn.Module):  # 1. Herde de torch.nn.Module
    def __init__(self, albumentations_transform: A.Compose):
        super().__init__()  # 2. Inicialize o Module
        self.albumentations_transform = albumentations_transform

    def forward(self, inpt: Any) -> Any:  # 3. Renomeie __call__ para forward
        is_tv_image = isinstance(inpt, tv_tensors.Image)
        
        if is_tv_image:
            image_tensor = inpt.as_subclass(torch.Tensor)
        elif isinstance(inpt, torch.Tensor):
            image_tensor = inpt
        else:
            return inpt

        if image_tensor.dim() != 3:
            return inpt

        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        augmented = self.albumentations_transform(image=img_np)
        img_augmented_np = augmented['image']
        result_tensor = torch.from_numpy(img_augmented_np).permute(2, 0, 1).to(image_tensor.device)

        if is_tv_image:
            return tv_tensors.Image(result_tensor)
        else:
            return result_tensor

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

    # def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
    #     npy_path, label = self.samples[idx]
    #     arr = np.load(npy_path)
        
    #     # Garante formato correto (C, H, W)
    #     if arr.ndim == 2:  # Grayscale
    #         arr = np.expand_dims(arr, axis=0)
    #     elif arr.ndim == 3 and arr.shape[0] not in [1, 3]:  # HWC -> CHW
    #         arr = np.transpose(arr, (2, 0, 1))  
        
    #     if self.transform:
    #         arr = self.transform(arr)

    #     tensor = torch.from_numpy(arr).float()
            
    #     return tensor, label

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        arr = np.load(npy_path).astype(np.float32)
        
        # Se estiver em HWC, move o C para frente automaticamente
        # A lógica: se o último eixo for o menor, provavelmente é o canal
        if arr.ndim == 3:
            if arr.shape[2] < arr.shape[0] and arr.shape[2] < arr.shape[1]:
                arr = np.transpose(arr, (2, 0, 1))
                
        tensor = torch.from_numpy(arr)
        # tv_tensors.Image é o padrão ouro da API v2, recomendo manter!
        image = tv_tensors.Image(tensor)

        if self.transform:
            image = self.transform(image)
            
        return image, label


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


# def load_datasets(batch_size: int, config, model_config) -> Tuple[DataLoader, DataLoader, int, torch.Tensor, List[str], np.ndarray]:
#     """
#     Prepara DataLoaders para treino e validação com amostragem balanceada.
    
#     Retorna:
#         Tuple[DataLoader, DataLoader, int, torch.Tensor, List[str]]: 
#         (train_loader, val_loader, num_classes, class_weights, class_names)
#     """

#     # Carrega dataset completo
#     full_dataset = NPYFolderDataset(root_dir=config.data_dir)
#     num_classes = len(full_dataset.classes)
#     class_names = full_dataset.classes

#     temp_loader = DataLoader(
#         full_dataset,
#         batch_size=Config.batch_size,
#         num_workers=config.num_workers,
#         shuffle=False
#     )

#     norm_mean, norm_std = compute_stats(temp_loader, full_dataset)

#     # Calcula pesos de classe
#     targets = [label for _, label in full_dataset.samples]
#     class_weights = compute_class_weights(targets, num_classes)
#     sample_weights = class_weights[torch.tensor(targets, dtype=torch.long)]

#     # Divide em treino/validação
#     total = len(full_dataset)
#     val_size = int(0.2 * total)
#     train_size = total - val_size
#     train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

#     # Aplica transformações específicas
#     if config.train_transform:
#         train_dataset = TransformSubset(train_subset, transform=config.train_transform)
#     else:
#         train_dataset = TransformSubset(
#             train_subset,
#             v2.Compose(
#                 [v2.ToDtype(torch.float32), v2.Normalize(mean=norm_mean, std=norm_std)]
#             ),
#         )

#     if config.val_transform:
#         val_dataset = TransformSubset(val_subset, transform=config.val_transform)
#     else:
#         val_dataset = TransformSubset(
#             val_subset,
#             v2.Compose(
#                 [v2.ToDtype(torch.float32), v2.Normalize(mean=norm_mean, std=norm_std)]
#             ),
#         )

#     # Configura amostrador balanceado
#     train_sampler = WeightedRandomSampler(
#         weights=sample_weights[train_subset.indices],
#         num_samples=len(train_subset),
#         replacement=True
#     )

#     # Augmentation
#     image_transforms, batch_transforms = get_augmentations(model_config, num_classes)

#     # Aplicar image-level transforms no dataset
#     if image_transforms:
#         train_transform = v2.Compose(image_transforms)
#         train_dataset = TransformSubset(train_subset, transform=train_transform)

#     # Cria transformação combinada de batch-level com random choice
#     if batch_transforms:
#         cutmix_or_mixup = v2.RandomChoice(batch_transforms)
#         def collate_fn(batch):
#             return cutmix_or_mixup(*default_collate(batch))
#     else:
#         collate_fn = None


#     # Cria DataLoaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         sampler=train_sampler,
#         num_workers=config.num_workers,
#         pin_memory=False,
#         collate_fn=collate_fn  
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=config.num_workers,
#         pin_memory=False
#     )

#     sample_input, _ = full_dataset[0]
#     input_example = sample_input.unsqueeze(0).to(Config.device).cpu().numpy().astype(np.float32)
#     input_schema = infer_signature(input_example)

#     return train_loader, val_loader, num_classes, class_weights, class_names, input_schema


def load_datasets(batch_size: int, config, generator: torch.Generator) -> Tuple[DataLoader, DataLoader, int, int, torch.Tensor, List[str], np.ndarray]:
    """
    Prepara DataLoaders para treino e validação com amostragem balanceada.
    Integra configurações de augmentação dinâmica do Optuna.
    
    Args:
        batch_size: Tamanho do batch
        config: Instância de Config ou DynamicConfig com configurações gerais e de augmentação
        generator: Gerador para reprodutibilidade
    
    Returns:
        Tuple[DataLoader, DataLoader, int, int, torch.Tensor, List[str], np.ndarray]: 
        (train_loader, val_loader, num_classes, input_channels, class_weights, class_names, input_schema)
    """

    # Carrega dataset completo
    full_dataset = NPYFolderDataset(root_dir=config.data_dir)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    input_channels = getattr(config, 'input_channels', 3) # Assumes config.input_channels is set

    # Loader temporário para calcular estatísticas
    temp_loader = DataLoader(
        full_dataset,
        batch_size=batch_size, # Use a batch_size normal
        num_workers=config.num_workers,
        shuffle=False
    )

    norm_mean, norm_std = compute_stats(temp_loader, full_dataset)

    # Define transformações base para validação (redimensionar e normalizar)
    # E fallback para treino se não houver augmentação dinâmica
    base_transforms = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), # Common size, adjust if needed
        v2.ToDtype(torch.float32), 
        v2.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    # === Lógica de Augmentação Dinâmica ===
    train_per_image_pipeline = None
    train_collate_fn = None
    
    augmentation_config = getattr(config, 'augmentation_config', {})
    
    if augmentation_config.get("use_augmentation", False):
        per_image_aug_method = augmentation_config.get("per_image_aug_method", "none")
        batch_level_aug_method = augmentation_config.get("batch_level_aug_method", "none")
        
        alb_transform = None
        if per_image_aug_method == "basic":
            alb_transform = AugmentationManager.get_augmentation_pipeline(
                "basic", **augmentation_config.get("basic_params", {})
            )
        elif per_image_aug_method == "advanced":
            alb_transform = AugmentationManager.get_augmentation_pipeline(
                "advanced", **augmentation_config.get("advanced_params", {})
            )
        
        if alb_transform:
            train_per_image_pipeline = v2.Compose([
                AlbumentationsWrapper(alb_transform), # Wrap Albumentations here
                v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC), # Ensure consistent size after augs
                v2.ToDtype(torch.float32), 
                v2.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            # If no specific per-image aug chosen, but use_augmentation is True, fallback to base transforms
            train_per_image_pipeline = base_transforms
            
        if batch_level_aug_method == "cutmix":
            train_collate_fn = CutMixWrapper(
                num_classes=num_classes, 
                alpha=augmentation_config.get("mix_alpha", 1.0)
            )
        elif batch_level_aug_method == "mixup":
            train_collate_fn = MixUpWrapper(
                num_classes=num_classes, 
                alpha=augmentation_config.get("mix_alpha", 0.2)
            )            
    # If no augmentation is used at all, train_per_image_pipeline should be base_transforms
    if train_per_image_pipeline is None:
        train_per_image_pipeline = base_transforms


    # Calcula pesos de classe
    targets = [label for _, label in full_dataset.samples]
    class_weights = compute_class_weights(targets, num_classes)
    sample_weights = class_weights[torch.tensor(targets, dtype=torch.long)]

    # Divide em treino/validação
    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Aplica transformações aos subsets
    train_dataset = TransformSubset(train_subset, transform=train_per_image_pipeline)
    val_dataset = TransformSubset(val_subset, transform=base_transforms) # Val sempre usa transforms base
    
    # Configura amostrador balanceado
    train_sampler = WeightedRandomSampler(
        weights=sample_weights[train_subset.indices],
        num_samples=len(train_subset),
        replacement=True
    )

    # Cria DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn= (lambda batch: train_collate_fn(default_collate(batch))) if train_collate_fn else default_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False
    )

    # Cria input schema para MLflow
    # Apply base transforms to sample_input before unsqueeze and infer_signature
    processed_sample_input, _ = val_dataset[0] 
    input_example = processed_sample_input.unsqueeze(0).to(config.device).cpu().numpy().astype(np.float32)
    input_schema = infer_signature(input_example)

    return train_loader, val_loader, num_classes, input_channels, class_weights, class_names, input_schema


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