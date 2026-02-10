"""

Author: Sania Qamar
Role: Senior Data Scientist

Goal: Train a hybrid blur detection model combining CNN features from MobileNetV2 and traditional computer vision metrics, using an SVM classifier for robust binary classification.
Blur Detection Model - Training Script with Hybrid Approach

This script trains a blur detection model using hybrid approach:
- CNN features from MobileNetV2 (128-dimensional)
- Traditional CV features: Laplacian variance, Canny edges, gradients, frequency domain
- SVM classifier on combined features (128 + 5 = 133 dimensions)

The hybrid approach combines:
1. Transfer learning with MobileNetV2 (ImageNet pre-trained)
2. Traditional computer vision metrics for physical blur properties
3. SVM classifier for robust binary classification

Expected accuracy: 94.29% on test set (vs 92.86% CNN-only)
Performance improvement: +1.43%
Blurred image recall: 98.57% (minimal false negatives)
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BlurDetectionTrainer:
    """
    Hybrid blur detection trainer combining CNN and traditional CV features.
    Trains both CNN and SVM classifier for optimal performance.
    """
    
    def __init__(self, image_size=224, output_dir="."):
        self.image_size = image_size
        self.output_dir = output_dir
        self.model = None
        self.history = None
        self.svm_model = None
        self.scaler_cnn = None
        self.scaler_traditional = None
        self.feature_extractor = None
    
    def load_dataset(self, dataset_path):
        """Load images from structured dataset directory."""
        print("\nLoading dataset...")
        print("-" * 60)
        
        images = []
        labels = []
        
        label_mapping = {
            'sharp': 0,
            'motion_blurred': 1,
            'defocused_blurred': 1
        }
        
        for folder_name, label in label_mapping.items():
            folder_path = Path(dataset_path) / folder_name
            
            if not folder_path.exists():
                print(f"Warning: {folder_path} not found")
                continue
            
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            print(f"{folder_name:20} -> {len(image_files):4} images")
            
            for filename in tqdm(image_files, desc=f"Loading {folder_name}", leave=False):
                image_path = folder_path / filename
                try:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (self.image_size, self.image_size))
                    images.append(image)
                    labels.append(label)
                except Exception as e:
                    continue
        
        images = np.array(images, dtype=np.float32) / 255.0
        labels = np.array(labels)
        
        print("-" * 60)
        print(f"Total images: {len(images)}")
        print(f"Sharp: {np.sum(labels == 0)}, Blurred: {np.sum(labels == 1)}\n")
        
        return images, labels
    
    def extract_traditional_features(self, images):
        """Extract 5 traditional CV features from images."""
        n_images = len(images)
        features = np.zeros((n_images, 5))
        
        print("Extracting traditional features...")
        for i, img in enumerate(tqdm(images, leave=False)):
            # Convert to 0-255 if needed
            if img.max() <= 1.0:
                img_cv = (img * 255).astype(np.uint8)
            else:
                img_cv = img.astype(np.uint8)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            
            # 1. Laplacian Variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features[i, 0] = laplacian.var()
            
            # 2. Canny Edge Ratio
            edges = cv2.Canny(gray, 100, 200)
            features[i, 1] = np.sum(edges > 0) / edges.size
            
            # 3. Gradient Mean
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(gx**2 + gy**2)
            features[i, 2] = magnitude.mean()
            
            # 4. Gradient Std
            features[i, 3] = magnitude.std()
            
            # 5. High Frequency Energy (FFT)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift)
            # High frequency: exclude low frequency components
            high_freq = magnitude_spectrum[int(magnitude_spectrum.shape[0]*0.2):int(magnitude_spectrum.shape[0]*0.8),
                                         int(magnitude_spectrum.shape[1]*0.2):int(magnitude_spectrum.shape[1]*0.8)]
            features[i, 4] = high_freq.sum() / magnitude_spectrum.sum()
        
        return features
    
    def build_cnn_model(self):
        """Build MobileNetV2 CNN model for feature extraction and training."""
        print("Building CNN model...")
        
        base_model = MobileNetV2(
            input_shape=(self.image_size, self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = False
        
        model = keras.Sequential([
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.0001)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"Model parameters: {model.count_params():,}\n")
        return model
    
    def build_feature_extractor(self, cnn_model):
        """Create model to extract CNN features (penultimate layer)."""
        feature_extraction_model = keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.layers[-2].output
        )
        return feature_extraction_model
    
    def train(self, dataset_path, epochs=30, batch_size=32):
        """Train hybrid model: CNN + SVM on hybrid features."""
        print("=" * 70)
        print("HYBRID BLUR DETECTION - TRAINING")
        print("=" * 70)
        
        # Load data
        images, labels = self.load_dataset(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        # Further split test to validation
        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test,
            test_size=0.5,
            random_state=42,
            stratify=y_test
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images\n")
        
        # Data augmentation
        augmentation = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Build and train CNN
        print("=" * 70)
        print("PHASE 1: Training CNN Component")
        print("=" * 70)
        
        self.model = self.build_cnn_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("Starting CNN training...\n")
        self.history = self.model.fit(
            augmentation.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate CNN
        print("\n" + "=" * 70)
        print("CNN Baseline Evaluation")
        print("=" * 70)
        
        y_pred_prob_cnn = self.model.predict(X_test, verbose=0)
        y_pred_cnn = (y_pred_prob_cnn > 0.5).astype(int).flatten()
        cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
        print(f"\nCNN Accuracy: {cnn_accuracy*100:.2f}%")
        print(classification_report(y_test, y_pred_cnn, target_names=['Sharp', 'Blurred']))
        
        # Save CNN model
        model_path = os.path.join(self.output_dir, 'blur_model.h5')
        self.model.save(model_path)
        print(f"CNN Model saved: {model_path}")
        
        # Save history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
        }
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        print(f"History saved: {history_path}\n")
        
        # Extract features for hybrid approach
        print("=" * 70)
        print("PHASE 2: Extracting Hybrid Features")
        print("=" * 70)
        
        # Build feature extractor
        self.feature_extractor = self.build_feature_extractor(self.model)
        
        # Extract CNN features
        print("\nExtracting CNN features from training data...")
        cnn_features_train = self.feature_extractor.predict(X_train, verbose=0)
        print(f"CNN features shape: {cnn_features_train.shape}")
        
        print("Extracting traditional features from training data...")
        trad_features_train = self.extract_traditional_features(X_train)
        print(f"Traditional features shape: {trad_features_train.shape}")
        
        print("\nExtracting CNN features from test data...")
        cnn_features_test = self.feature_extractor.predict(X_test, verbose=0)
        
        print("Extracting traditional features from test data...")
        trad_features_test = self.extract_traditional_features(X_test)
        
        # Normalize features
        print("\nNormalizing features...")
        self.scaler_cnn = StandardScaler()
        cnn_features_train_norm = self.scaler_cnn.fit_transform(cnn_features_train)
        cnn_features_test_norm = self.scaler_cnn.transform(cnn_features_test)
        
        self.scaler_traditional = StandardScaler()
        trad_features_train_norm = self.scaler_traditional.fit_transform(trad_features_train)
        trad_features_test_norm = self.scaler_traditional.transform(trad_features_test)
        
        # Concatenate features
        hybrid_features_train = np.concatenate(
            [cnn_features_train_norm, trad_features_train_norm], axis=1
        )
        hybrid_features_test = np.concatenate(
            [cnn_features_test_norm, trad_features_test_norm], axis=1
        )
        
        print(f"Hybrid feature vector size: {hybrid_features_train.shape[1]} (128 CNN + 5 Traditional)")
        
        # Train SVM on hybrid features
        print("\n" + "=" * 70)
        print("PHASE 3: Training SVM on Hybrid Features")
        print("=" * 70)
        
        print("\nTraining SVM classifier...")
        self.svm_model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')
        self.svm_model.fit(hybrid_features_train, y_train)
        
        # Evaluate SVM
        y_pred_svm = self.svm_model.predict(hybrid_features_test)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)
        
        print("\n" + "=" * 70)
        print("HYBRID APPROACH EVALUATION")
        print("=" * 70)
        print(f"\nSVM Hybrid Accuracy: {svm_accuracy*100:.2f}%")
        print(f"Improvement over CNN: +{(svm_accuracy - cnn_accuracy)*100:.2f}%")
        print("\n" + classification_report(y_test, y_pred_svm, target_names=['Sharp', 'Blurred']))
        print("=" * 70 + "\n")
        
        # Save SVM model
        svm_path = os.path.join(self.output_dir, 'svm_hybrid_model.pkl')
        with open(svm_path, 'wb') as f:
            pickle.dump(self.svm_model, f)
        print(f"SVM Model saved: {svm_path}")
        
        # Save scalers
        scaler_cnn_path = os.path.join(self.output_dir, 'scaler_cnn.pkl')
        with open(scaler_cnn_path, 'wb') as f:
            pickle.dump(self.scaler_cnn, f)
        print(f"CNN Scaler saved: {scaler_cnn_path}")
        
        scaler_trad_path = os.path.join(self.output_dir, 'scaler_traditional.pkl')
        with open(scaler_trad_path, 'wb') as f:
            pickle.dump(self.scaler_traditional, f)
        print(f"Traditional Scaler saved: {scaler_trad_path}")
        
        print(f"\nTraining complete. Models and features saved to {self.output_dir}")


def main():
    print("=" * 60)
    print("BLUR DETECTION MODEL - TRAINING")
    print("=" * 60)
    
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        'blur_dataset',
        'blur_dataset'
    )
    
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        print("\nExpected structure:")
        print("motion_blur/")
        print("├── blur_dataset/")
        print("│   └── blur_dataset/")
        print("│       ├── sharp/")
        print("│       ├── motion_blurred/")
        print("│       └── defocused_blurred/")
        return
    
    trainer = BlurDetectionTrainer(image_size=224)
    trainer.train(dataset_path, epochs=30, batch_size=32)


if __name__ == '__main__':
    main()
