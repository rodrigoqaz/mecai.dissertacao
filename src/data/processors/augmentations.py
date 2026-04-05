import cv2
import numpy as np
import albumentations as A
from torchvision.transforms import v2

class AugmentationManager:
    @staticmethod
    def get_augmentation_pipeline(method: str, **params):
        if method == "basic":
            return A.Compose([

                A.OneOf([
                    A.Rotate(limit=params.get('rotation_range', 20), p=0.5),
                    A.HorizontalFlip(p=params.get('horizontal_flip_prob', 0.5)),
                    A.VerticalFlip()

                ])

                ,
                A.RandomBrightnessContrast(
                    brightness_limit=params.get('brightness_range', 0.1),
                    contrast_limit=params.get('contrast_range', 0.1),
                    p=0.3
                ),
                
                A.Affine(
                    translate_percent=params.get('width_shift_range', 0.1),
                    scale=(0.8, 1.2),
                    rotate=(-15, 15),
                    shear=(-10, 10),
                    interpolation=cv2.INTER_LINEAR,
                    fit_output=True,
                    p=0.7
                ),

                A.Resize(224, 224)
            ])
        
        elif method == "advanced":
            p_distort = params.get('p_distort', 0.3)
            p_noise = params.get('p_noise', 0.2)
            p_dropout = params.get('p_dropout', 0.2)
            num_holes = params.get('num_holes', 8)
            crop_min = params.get('crop_scale_min', 0.7)

            return A.Compose([
                # 1. Geometria Rígida (Sempre útil)
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Transpose(p=0.5),

                # 2. Invariância de Escala (Simula Zoom e muda enquadramento)
                A.RandomResizedCrop(
                    size=(224, 224), 
                    scale=(crop_min, 1.0), 
                    ratio=(0.8, 1.25), 
                    p=1.0
                ),

                # 3. Distorções Elásticas e Ópticas (Controladas por p_distort)
                # OneOf garante que não aplicamos todas ao mesmo tempo (destrutivo)
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, p=1.0
                    ),
                    A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    A.OpticalDistortion(distort_limit=0.5, p=1.0),
                ], p=p_distort),

                # 4. Ruído e Intensidade (Controladas por p_noise)
                # Seguro para 15 canais (evita mexer em Hue/Saturation)
                A.OneOf([
                    A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1.0
                    ),
                ], p=p_noise),

                # 5. Regularização por Oclusão (CoarseDropout)
                # Força o modelo a aprender contexto global
                A.CoarseDropout(
                    num_holes_range=(2, num_holes),
                    hole_height_range=(8, 32),
                    hole_width_range=(8, 32),
                    fill=0,
                    p=p_dropout
                ),

                # Garantia final de tamanho
                A.Resize(height=224, width=224)
            ])
        return None

    @staticmethod
    def apply_augmentation(image: np.ndarray, method: str, n_augments: int = 1, **params) -> np.ndarray:
        """Retorna lista com N imagens aumentadas + original"""
        pipeline = AugmentationManager.get_augmentation_pipeline(method, **params)
        augmented_images = [image]
        for _ in range(n_augments):
            augmented = pipeline(image=image)
            augmented_images.append(augmented['image'])
            
        return augmented_images



class CutMixWrapper:
    def __init__(self, num_classes, alpha=1.0):
        self.transform = v2.CutMix(num_classes=num_classes, alpha=alpha)
    
    def __call__(self, batch):
        return self.transform(*batch)

class MixUpWrapper:
    def __init__(self, num_classes, alpha=0.2):
        self.transform = v2.MixUp(num_classes=num_classes, alpha=alpha)
    
    def __call__(self, batch):
        return self.transform(*batch)
