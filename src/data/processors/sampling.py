import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import cv2
import random
from typing import Dict, List, Tuple
from .augmentations import AugmentationManager


class SamplingManager:
    @staticmethod
    def apply_sampling(
        processed_images: Dict[str, List[Tuple[np.ndarray, str]]],
        method: str,
        **kwargs
    ) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        """
        Aplica técnica de amostragem e retorna novo dicionário de imagens
        Formato de entrada/saída: {class_name: [(image_array, filename)]}
        """
        if method == "smote":
            return SamplingManager._apply_smote(processed_images, **kwargs)
        elif method == "undersample":
            return SamplingManager._apply_undersample(processed_images, **kwargs)
        elif method == "oversample":
            return SamplingManager._apply_oversample(processed_images, **kwargs)
        elif method == "oversample_with_augmentation":
            return SamplingManager._apply_oversample_with_augmentation(processed_images, **kwargs)
        else:
            raise ValueError(f"Método não suportado: {method}")

    @staticmethod
    def _apply_smote(
        processed_images: Dict[str, List[Tuple[np.ndarray, str]]],
        k_neighbors: int = 5,
        sampling_strategy: str = 'auto'
    ) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        # Preparar dados para SMOTE
        features, labels, original_filenames = SamplingManager._prepare_smote_data(processed_images)
        
        # Aplicar SMOTE
        sm = SMOTE(
            k_neighbors=min(k_neighbors, len(features)-1),
            sampling_strategy=sampling_strategy,
            random_state=42
        )
        X_res, y_res = sm.fit_resample(features, labels)
        
        # Gerar novas imagens sintéticas
        return SamplingManager._create_synthetic_images(X_res, y_res, processed_images, original_filenames)

    @staticmethod
    def _prepare_smote_data(processed_images):
        features = []
        labels = []
        label_encoder = LabelEncoder()
        original_filenames = {}
        
        # Verificar formato das imagens
        first_img, _ = processed_images[list(processed_images.keys())[0]][0]
        is_grayscale = len(first_img.shape) == 2 or first_img.shape[2] == 1
        
        for class_name, images in processed_images.items():
            original_filenames[class_name] = [filename for _, filename in images]
            for img, _ in images:
                # Converter para escala de cinza se necessário
                if is_grayscale and len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Achatar mantendo a consistência de canais
                features.append(img.flatten())
                labels.append(class_name)
        
        return np.array(features), label_encoder.fit_transform(labels), original_filenames

    @staticmethod
    def _standardize_image(img: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
        # Converter para escala de cinza se necessário
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Redimensionar
        img = cv2.resize(img, target_size)
        
        # Adicionar canal se for grayscale
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Normalizar
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _create_synthetic_images(X_res, y_res, original_images, original_filenames):
        synthetic_images = defaultdict(list)
        label_encoder = LabelEncoder()
        label_encoder.fit(list(original_images.keys()))
        decoded_labels = label_encoder.inverse_transform(y_res)
        
        # Obter formato da primeira imagem original
        first_img, _ = original_images[list(original_images.keys())[0]][0]
        target_shape = first_img.shape  # (altura, largura, canais)
        
        # Verificar consistência de canais
        total_pixels = target_shape[0] * target_shape[1]
        if len(target_shape) == 3:
            total_pixels *= target_shape[2]
        
        for features, label in zip(X_res, decoded_labels):
            try:
                # Redimensionar corretamente
                if features.size != total_pixels:
                    raise ValueError(f"Tamanho incompatível: {features.size} vs {total_pixels}")
                
                # Reshape considerando canais
                if len(target_shape) == 3:
                    img = features.reshape(target_shape)
                else:
                    img = features.reshape(target_shape)
                    img = np.expand_dims(img, axis=-1)  # Adicionar canal se necessário
                    
                # Converter para uint8 e escala 0-255
                img = (img * 255).astype(np.uint8)
                
                # Se necessário, converter grayscale para RGB
                if img.shape[-1] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Gerar nome único
                original_filename = original_filenames[label][0][:-4]  # Pega o nome sem extensão
                filename = f"{original_filename}_smote_{len(synthetic_images[label])}.png"
                synthetic_images[label].append((img, filename))
                
            except Exception as e:
                print(f"Erro ao gerar imagem: {str(e)}")
                continue
                
        return synthetic_images

    @staticmethod
    def _apply_undersample(
        processed_images: Dict[str, List[Tuple[np.ndarray, str]]],
        ratio: float = 1.0
    ) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        min_samples = int(min(len(imgs) for imgs in processed_images.values()) * ratio)
        return {
            cls: random.sample(imgs, min_samples)
            for cls, imgs in processed_images.items()
        }

    @staticmethod
    def _apply_oversample(
        processed_images: Dict[str, List[Tuple[np.ndarray, str]]],
        ratio: float = 1.0
    ) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        max_samples = int(max(len(imgs) for imgs in processed_images.values()) * ratio)
        return {
            cls: SamplingManager._oversample_class(imgs, max_samples)
            for cls, imgs in processed_images.items()
        }

    @staticmethod
    def _apply_oversample_with_augmentation(
        processed_images: Dict[str, List[Tuple[np.ndarray, str]]],
        ratio: float = 1.0,
        augmentation_method: str = "basic",
        **kwargs
    ) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        max_samples = int(max(len(imgs) for imgs in processed_images.values()) * ratio)
        return {
            cls: SamplingManager._oversample_class_with_augmentation(
                imgs, max_samples, augmentation_method, **kwargs
            )
            for cls, imgs in processed_images.items()
        }

    @staticmethod
    def _oversample_class(images: list, target_count: int) -> list:
        new_images = []
        counters = dict()

        while len(new_images) < target_count:
            needed = target_count - len(new_images)
            batch = random.choices(images, k=needed)
            for img, filename in batch:
                name, ext = os.path.splitext(filename)
                if name not in counters:
                    counters[name] = 1
                else:
                    counters[name] += 1
                new_filename = f"{name}_aug_{counters[name]:02d}{ext}"
                new_images.append((img, new_filename))
                if len(new_images) == target_count:
                    break
        return new_images

    @staticmethod
    def _oversample_class_with_augmentation(
        images: list, target_count: int, augmentation_method: str, **params
    ) -> list:
        original_count = len(images)
        if original_count >= target_count:
            return images[:target_count]
        
        new_images = list(images)
        aug_counters = {}
        
        for _ in range(target_count - original_count):
            base_image, filename = random.choice(images)
            
            if filename not in aug_counters:
                aug_counters[filename] = 0
            aug_counters[filename] += 1
            
            augmented = AugmentationManager.apply_augmentation(
                base_image, 
                method=augmentation_method,
                n_augments=1,
                **params
            )[1]
            
            new_filename = f"{os.path.splitext(filename)[0]}_aug_{aug_counters[filename]:04d}.npy"
            new_images.append((augmented, new_filename))
        
        return new_images