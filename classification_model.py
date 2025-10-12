"""
Classification Model for Breast Ultrasound Project
Simple model class for medical image classification
"""

import numpy as np
import cv2
import tensorflow as tf
from typing import Dict, Tuple, Optional
import os

class ClassificationModel:
    """class to classify breast ultrasound images"""
    def __init__(self, model_path: str = None):
        """
        Set up the classification model
        Args:
            model_path: Path to the trained model file (.h5 format)
        """
        self.model = None
        self.model_path = model_path
        self.input_size = (224, 224) 
        self.classes = ['benign', 'malignant']  # 2 classification types: benign or malignant
        self.is_model_loaded = False
        
    def load_model(self, model_path: str = None) -> bool:
        """
        Load the trained model for making predictions
        Args:
            model_path: Path to model file (optional if already set)
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Use provided path or the one set earlier
            path = model_path or self.model_path
            
            if not path or not os.path.exists(path):
                print(f"Cannot find model file: {path}")
                return False
                
            # Load the trained Keras model
            self.model = tf.keras.models.load_model(path)
            self.is_model_loaded = True
            print(f"Model loaded successfully from: {path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_model_loaded = False
            return False
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepare image for the model - accepts any size image
        Args:
            image: Input image as numpy array (any dimensions)
        Returns:
            Processed image ready for prediction or None if error
        """
        try:
            # Handle different image formats and convert to RGB
            if len(image.shape) == 2:
                # Grayscale image - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Single channel image - convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # BGR image - convert to RGB (OpenCV uses BGR format)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA image - remove alpha channel and convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            # Resize from any input size to model's required size (224x224)
            image = cv2.resize(image, self.input_size)
            
            # Normalize pixel values to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def predict(self, image: np.ndarray) -> Dict[str, any]:
        """
        Classify the ultrasound image - accepts any size image
        Args:
            image: Input ultrasound image as numpy array (any size)
        Returns:
            Dictionary with prediction results
        """
        try:
            # Check if model is ready
            if not self.is_model_loaded or self.model is None:
                return None
            
            # Process the image first
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None
            # Run the prediction
            predictions = self.model.predict(processed_image, verbose=0)
            # Get the probabilities for each class
            probabilities = predictions[0]
            # Find which class has highest probability
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.classes[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])
            # Make dictionary with all probabilities
            class_probabilities = {
                class_name: float(prob) 
                for class_name, prob in zip(self.classes, probabilities)
            }
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': class_probabilities,
                'success': True
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    

    
    def get_class_names(self) -> list:
        """
        Get list of class names
        Returns:
            List of class names
        """
        return self.classes.copy()
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the model
        Returns:
            Dictionary with model information
        """
        return {
            'model_loaded': self.is_loaded(),
            'model_path': self.model_path,
            'input_size': self.input_size,
            'classes': self.classes,
            'num_classes': len(self.classes)
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the classification model
    print("Testing Classification Model...")
    # Initialize model (without loading)
    classifier = ClassificationModel()
    # Test demo mode with different image sizes
    print("\n--- Demo Mode Test with Various Image Sizes ---")
    
    # Test different image sizes and formats
    test_images = [
        np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8),  # Small RGB
        np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8),  # Large RGB
        np.random.randint(0, 255, (256, 256), dtype=np.uint8),     # Grayscale
        np.random.randint(0, 255, (1024, 768, 4), dtype=np.uint8), # RGBA
    ]
    
    for i, demo_image in enumerate(test_images):
        print(f"Testing image {i+1} with shape: {demo_image.shape}")
        result = classifier.predict(demo_image)
        print(f"Result: {result['prediction']} (confidence: {result['confidence']:.2f})")
    
    # Test model info
    print(f"\nModel Info: {classifier.get_model_info()}")
    
    # Test with actual model path (if it exists)
    model_path = "../breast_cancer_model_no_finetune.h5"
    if os.path.exists(model_path):
        print(f"\n--- Real Model Test ---")
        classifier = ClassificationModel(model_path)
        if classifier.load_model():
            # Test with first image
            result = classifier.predict(test_images[0])
            print(f"Real Model Result: {result}")
    else:
        print(f"\nModel file not found at: {model_path}")
        print("Place your trained model file to test with real predictions")