import cv2
import numpy as np
import logging
import os
from typing import Dict, List, Tuple
from src.data.processors import augmentations, features, sampling

class ImageUtils:
    @staticmethod
    def resize(image: np.ndarray, size: tuple) -> np.ndarray:
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
        gray = ImageUtils.rgb_to_grayscale(image)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
        gray = ImageUtils.rgb_to_grayscale(image)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gabor_filter(image: np.ndarray, frequency: float, theta: float, bandwidth: float) -> np.ndarray:
        gray = ImageUtils.rgb_to_grayscale(image)
        kernel = cv2.getGaborKernel(
            (21, 21),
            bandwidth,
            np.deg2rad(theta),
            frequency,
            0.5,
            0,
            ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)


class ImageProcessor:
    """
    Responsável por aplicar transformações às imagens.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def process_dataset(self, dataset: Dict[str, List]) -> Dict[str, List]:
        """
        Aplica todas as etapas de processamento configuradas na ordem definida.
        """
        for step in self.config_manager.get_processing_steps():
            step_type = step['type']
            
            if step_type == 'resize':
                dataset = self._apply_resize(dataset, step['params'])
            elif step_type == 'augment':
                dataset = self._apply_augmentation(dataset, step['params'])
            elif step_type == 'feature':
                dataset = self._apply_feature_engineering(dataset, step['params'])
            elif step_type == 'sampling':
                dataset = self._apply_sampling(dataset, step['params'])
                
        return dataset
        
    def _apply_resize(self, dataset, params) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        logging.info(f"Redimensionando as imagens: {params['size']}")
        resized_dataset = {}
        for class_name, images in dataset.items():
            resized_dataset[class_name] = []
            for img, filename in images:
                resized_img = ImageUtils.resize(img, params['size'])
                resized_dataset[class_name].append((resized_img, filename))
        return resized_dataset
        
    def _apply_augmentation(self, dataset, params) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        logging.info(f"Aumentando as imagens. Método: {params['method']}")
        augmented_dataset = {}
        for class_name, images in dataset.items():
            augmented_dataset[class_name] = []
            for img, filename in images:
                augmented_imgs = augmentations.AugmentationManager.apply_augmentation(
                    img,
                    method=params['method'],
                    n_augments=params['args']['n_augments']
                )
                
                name, ext = os.path.splitext(filename)
                augmented_dataset[class_name].append((augmented_imgs[0], filename))
                for idx, aug_img in enumerate(augmented_imgs[1:], 1):
                    new_name = f"{name}_aug{idx:02d}{ext}"
                    augmented_dataset[class_name].append((aug_img, new_name))
                    
        total_classes = len(dataset)
        total_images_inicial = sum(len(imgs) for imgs in dataset.values())
        total_images_aumentadas = sum(len(imgs) for imgs in augmented_dataset.values())
        logging.info(f"Total de classes carregadas: {total_classes}")
        logging.info(f"Total de imagens carregadas: {total_images_inicial}")
        logging.info(f"Total de imagens aumentadas: {total_images_aumentadas}")
        return augmented_dataset
        
    def _apply_feature_engineering(self, dataset, params) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        logging.info(f"Aplicando Feature Engineering. Método: {params['method']}")
        featured_dataset = {}
        for class_name, images in dataset.items():
            featured_dataset[class_name] = []
            for img, filename in images:
                featured_img = features.apply_feature_engineering(
                    img, params["method"], **params["args"]
                )
                name, ext = os.path.splitext(filename)
                new_name = f"{name}_{params['method']}{ext}"
                featured_dataset[class_name].append((featured_img, new_name))
        return featured_dataset
        
    def _apply_sampling(self, dataset, params) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        logging.info(f"Aplicando Sampling. Método: {params['method']}")
        sampled_dataset =  sampling.SamplingManager.apply_sampling(
            dataset,
            params['method'],
            **params['args']
        )
        total_classes = len(dataset)
        total_images_inicial = sum(len(imgs) for imgs in dataset.values())
        total_images_amostradas = sum(len(imgs) for imgs in sampled_dataset.values())
        logging.info(f"Total de classes carregadas: {total_classes}")
        logging.info(f"Total de imagens carregadas: {total_images_inicial}")
        logging.info(f"Total de imagens amostradas: {total_images_amostradas}")
        return sampled_dataset
