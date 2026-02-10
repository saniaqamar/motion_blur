"""

Author: Sania Qamar
Role: Senior Data Scientist
Blur Detection Model - Inference Script with Hybrid Approach

Supports multiple models for blur detection:
- cnn: CNN-only baseline (92.86% accuracy)
- svm_hybrid: SVM on hybrid features (94.29% accuracy - RECOMMENDED)
- rf_hybrid: Random Forest on hybrid features (93.33% accuracy)
- ensemble: Weighted ensemble of models (93.81% accuracy)

Usage:
    python predict.py /path/to/images
    python predict.py /path/to/images --model svm_hybrid --output results.csv
"""

import os
import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import pickle
from sklearn.preprocessing import StandardScaler


class BlurPredictor:
    """Multi-model blur detection inference."""
    
    def __init__(self, model_type='svm_hybrid', model_dir='.'):
        """
        Initialize predictor with specified model.
        
        Args:
            model_type: Type of model [cnn, svm_hybrid, rf_hybrid, ensemble]
            model_dir: Directory containing model files
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.image_size = 224
        
        self.cnn_model = None
        self.svm_model = None
        self.rf_model = None
        self.feature_extractor = None
        self.scaler_cnn = None
        self.scaler_traditional = None
        
        self._load_models()
    
    def _load_models(self):
        """Load required models based on model_type."""
        print(f"\nLoading {self.model_type} model...")
        print("-" * 60)
        
        # Load CNN model (needed for all models)
        cnn_path = os.path.join(self.model_dir, 'blur_model.h5')
        if not os.path.exists(cnn_path):
            raise FileNotFoundError(
                f"CNN model not found: {cnn_path}\n"
                "Train the model first using: python train.py"
            )
        
        self.cnn_model = keras.models.load_model(cnn_path)
        print(f"CNN model loaded: {cnn_path}")
        
        # Create feature extractor (for hybrid models)
        if self.model_type in ['svm_hybrid', 'rf_hybrid', 'ensemble']:
            self.feature_extractor = keras.Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.layers[-2].output
            )
            
            # Load SVM model
            svm_path = os.path.join(self.model_dir, 'svm_hybrid_model.pkl')
            if not os.path.exists(svm_path):
                raise FileNotFoundError(f"SVM model not found: {svm_path}")
            with open(svm_path, 'rb') as f:
                self.svm_model = pickle.load(f)
            print(f"SVM model loaded: {svm_path}")
            
            # Load Random Forest model if needed
            if self.model_type in ['rf_hybrid', 'ensemble']:
                rf_path = os.path.join(self.model_dir, 'rf_hybrid_model.pkl')
                if os.path.exists(rf_path):
                    with open(rf_path, 'rb') as f:
                        self.rf_model = pickle.load(f)
                    print(f"Random Forest model loaded: {rf_path}")
            
            # Load scalers
            scaler_cnn_path = os.path.join(self.model_dir, 'scaler_cnn.pkl')
            scaler_trad_path = os.path.join(self.model_dir, 'scaler_traditional.pkl')
            
            if os.path.exists(scaler_cnn_path) and os.path.exists(scaler_trad_path):
                with open(scaler_cnn_path, 'rb') as f:
                    self.scaler_cnn = pickle.load(f)
                with open(scaler_trad_path, 'rb') as f:
                    self.scaler_traditional = pickle.load(f)
                print(f"Feature scalers loaded")
        
        print("-" * 60 + "\n")
    
    def _prepare_image(self, image_path):
        """Load and preprocess image for inference."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=0)
            
            return image
        except Exception:
            return None
    
    def _extract_traditional_features(self, image_array):
        """Extract 5 traditional CV features from image."""
        if image_array.max() <= 1.0:
            img_cv = (image_array * 255).astype(np.uint8)
        else:
            img_cv = image_array.astype(np.uint8)
        
        if img_cv.ndim == 3 and img_cv.shape[2] == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_cv
        
        features = np.zeros(5)
        
        # 1. Laplacian Variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features[0] = laplacian.var()
        
        # 2. Canny Edge Ratio
        edges = cv2.Canny(gray, 100, 200)
        features[1] = np.sum(edges > 0) / edges.size
        
        # 3. Gradient Mean
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        features[2] = magnitude.mean()
        
        # 4. Gradient Std
        features[3] = magnitude.std()
        
        # 5. High Frequency Energy (FFT)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        high_freq = magnitude_spectrum[int(magnitude_spectrum.shape[0]*0.2):int(magnitude_spectrum.shape[0]*0.8),
                                      int(magnitude_spectrum.shape[1]*0.2):int(magnitude_spectrum.shape[1]*0.8)]
        features[4] = high_freq.sum() / magnitude_spectrum.sum()
        
        return features.reshape(1, -1)
    
    def _predict_cnn(self, image):
        """Predict using CNN model."""
        confidence = float(self.cnn_model.predict(image, verbose=0)[0][0])
        prediction = 1 if confidence > 0.5 else 0
        return prediction, confidence
    
    def _predict_svm_hybrid(self, image):
        """Predict using SVM on hybrid features."""
        cnn_features = self.feature_extractor.predict(image, verbose=0)
        cnn_features_norm = self.scaler_cnn.transform(cnn_features)
        
        # Extract and normalize traditional features
        image_array = (image[0] * 255).astype(np.uint8) if image.max() <= 1.0 else image[0]
        trad_features = self._extract_traditional_features(image_array)
        trad_features_norm = self.scaler_traditional.transform(trad_features)
        
        # Concatenate
        hybrid_features = np.concatenate([cnn_features_norm, trad_features_norm], axis=1)
        
        # Predict
        confidence = self.svm_model.decision_function(hybrid_features)[0]
        confidence = 1.0 / (1.0 + np.exp(-confidence))  # Sigmoid normalization
        prediction = 1 if confidence > 0.5 else 0
        
        return prediction, confidence
    
    def _predict_rf_hybrid(self, image):
        """Predict using Random Forest on hybrid features."""
        if self.rf_model is None:
            return self._predict_svm_hybrid(image)
        
        cnn_features = self.feature_extractor.predict(image, verbose=0)
        cnn_features_norm = self.scaler_cnn.transform(cnn_features)
        
        image_array = (image[0] * 255).astype(np.uint8) if image.max() <= 1.0 else image[0]
        trad_features = self._extract_traditional_features(image_array)
        trad_features_norm = self.scaler_traditional.transform(trad_features)
        
        hybrid_features = np.concatenate([cnn_features_norm, trad_features_norm], axis=1)
        
        confidence = self.rf_model.predict_proba(hybrid_features)[0][1]
        prediction = 1 if confidence > 0.5 else 0
        
        return prediction, confidence
    
    def _predict_ensemble(self, image):
        """Predict using weighted ensemble of models."""
        pred_cnn, conf_cnn = self._predict_cnn(image)
        pred_svm, conf_svm = self._predict_svm_hybrid(image)
        
        if self.rf_model is not None:
            pred_rf, conf_rf = self._predict_rf_hybrid(image)
            confidence = 0.5 * conf_cnn + 0.25 * conf_svm + 0.25 * conf_rf
        else:
            confidence = 0.5 * conf_cnn + 0.5 * conf_svm
        
        prediction = 1 if confidence > 0.5 else 0
        return prediction, confidence
    
    def predict_image(self, image_path):
        """Predict blur status for single image."""
        prepared = self._prepare_image(image_path)
        if prepared is None:
            return None, None
        
        if self.model_type == 'cnn':
            prediction, confidence = self._predict_cnn(prepared)
        elif self.model_type == 'svm_hybrid':
            prediction, confidence = self._predict_svm_hybrid(prepared)
        elif self.model_type == 'rf_hybrid':
            prediction, confidence = self._predict_rf_hybrid(prepared)
        elif self.model_type == 'ensemble':
            prediction, confidence = self._predict_ensemble(prepared)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return prediction, min(max(confidence, 0.0), 1.0)
    
    def predict_folder(self, folder_path, output_csv='predictions.csv'):
        """Process all images in folder and save results to CSV."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        
        if not image_files:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Found {len(image_files)} images\n")
        
        results = []
        for filename in tqdm(image_files, desc="Processing", unit="image"):
            image_path = os.path.join(folder_path, filename)
            prediction, confidence = self.predict_image(image_path)
            
            if prediction is not None:
                results.append({
                    'filename': filename,
                    'prediction': prediction,
                    'confidence': round(confidence, 4),
                    'model_used': self.model_type
                })
        
        # Save to CSV
        if results:
            self._save_csv(results, output_csv)
        
        return results
    
    def _save_csv(self, results, output_path):
        """Save results to CSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'prediction', 'confidence', 'model_used'])
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {output_path}")
        
        # Print summary
        predictions = [r['prediction'] for r in results]
        sharp = sum(1 for p in predictions if p == 0)
        blurred = sum(1 for p in predictions if p == 1)
        total = len(predictions)
        
        print("\nSummary:")
        print("-" * 60)
        print(f"Total images: {total}")
        print(f"Sharp:   {sharp:4} ({100*sharp/total:5.1f}%)")
        print(f"Blurred: {blurred:4} ({100*blurred/total:5.1f}%)")
        print(f"Model: {self.model_type}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Predict blur status for images using hybrid model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py /path/to/images
  python predict.py /path/to/images --model svm_hybrid --output results.csv
  
Models:
  cnn         - CNN baseline (92.86% accuracy)
  svm_hybrid  - SVM on hybrid features (94.29% accuracy - RECOMMENDED)
  rf_hybrid   - Random Forest on hybrid features (93.33% accuracy)
  ensemble    - Weighted ensemble (93.81% accuracy)
        """
    )
    
    parser.add_argument('image_folder', type=str, help='Folder containing images to classify')
    parser.add_argument('--model', type=str, default='svm_hybrid', 
                       choices=['cnn', 'svm_hybrid', 'rf_hybrid', 'ensemble'],
                       help='Model to use (default: svm_hybrid)')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='Output CSV file path (default: predictions.csv)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BLUR DETECTION - INFERENCE")
    print("=" * 60)
    
    try:
        predictor = BlurPredictor(model_type=args.model)
        predictor.predict_folder(args.image_folder, args.output)
        print("\nDone.\n")
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
    except ValueError as e:
        print(f"\nError: {e}\n")
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")


if __name__ == '__main__':
    main()
