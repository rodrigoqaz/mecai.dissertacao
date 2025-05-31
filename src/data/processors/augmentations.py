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
            return A.Compose([
                A.RandomRotate90(),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=0.3),
                A.Resize(224, 224)
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
