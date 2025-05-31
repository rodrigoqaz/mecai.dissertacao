import cv2
import numpy as np

class FeatureEngineer:
    @staticmethod
    def apply_threshold(img, method='otsu', **params):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif method == 'adaptive':
            thresh = cv2.adaptiveThreshold(gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY,
                params.get('block_size', 11),
                params.get('c', 2))
        
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gabor(img, frequencies, thetas, bandwidth):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        responses = []
        
        for freq in frequencies:
            for theta in np.deg2rad(thetas):
                kernel = cv2.getGaborKernel(
                    (21, 21), 
                    bandwidth, 
                    theta, 
                    freq, 
                    0.5, 
                    0, 
                    ktype=cv2.CV_32F)
                
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                responses.append(filtered)
        
        return np.stack(responses, axis=-1)

def apply_feature_engineering(img, method, **params):
    fe = FeatureEngineer()
    
    if method == 'threshold+gabor':
        thresholded = fe.apply_threshold(img, **params.get('threshold', {}))
        gabor_features = fe.apply_gabor(img, **params.get('gabor', {}))
        return np.concatenate([thresholded, gabor_features], axis=-1)
    
    return img
