"""
ML Backend (Orchestrator) for Breast Ultrasound AI Analysis
Integrates classification and segmentation models
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, Any, List
import time
import os

# Import our model classes
from classification_model import ClassificationModel
from segmentation_model import SegmentationModel
from image_processor import ImageProcessor, ResultProcessor


class MLBackend:
    """Integrated ML backend for breast ultrasound analysis"""
    
    def __init__(self, classification_model_path: str = None, segmentation_model_path: str = None):
        """
        Initialize ML backend with both models
        Args:
            classification_model_path: Path to classification model
            segmentation_model_path: Path to segmentation model
        """
        # Initialize models
        self.classification_model = ClassificationModel(classification_model_path)
        self.segmentation_model = SegmentationModel(segmentation_model_path)
        
        # State tracking
        self.models_loaded = False
        self.last_analysis_time = 0
        
    def initialize(self) -> bool:
        """
        Initialize and load both models
        Returns:
            True if at least one model loaded successfully
        """
        try:
            print("Initializing ML Backend...")
            
            # Load classification model
            classification_loaded = False
            if self.classification_model.model_path:
                classification_loaded = self.classification_model.load_model()
            
            # Load segmentation model
            segmentation_loaded = False
            if self.segmentation_model.model_path:
                segmentation_loaded = self.segmentation_model.load_model()
            
            self.models_loaded = classification_loaded or segmentation_loaded
            
            if self.models_loaded:
                print("ML Backend initialized successfully")
            else:
                print("No models loaded - analysis will fail without models")
            
            return True
            
        except Exception as e:
            print(f"Error initializing ML backend: {e}")
            return False
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Complete analysis of ultrasound image
        Args:
            image_path: Path to the ultrasound image
        Returns:
            Dictionary containing analysis results
        """
        try:
            start_time = time.time()
            
            # Load image
            image = self._load_image(image_path)
            if image is None:
                return None
            
            # Run classification
            classification_result = self.classification_model.predict(image)
            
            # Run segmentation
            segmentation_result = self.segmentation_model.predict(image)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.last_analysis_time = processing_time
            
            # Combine results
            results = {
                'classification': classification_result,
                'segmentation': segmentation_result,
                'processing_time': processing_time,
                'image_path': image_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_shape': image.shape,
                'models_status': self.get_model_status()
            }
            
            # Process and format results using ResultProcessor
            formatted_results = ResultProcessor.format_results(
                classification_result, segmentation_result
            )
            
            # Merge with analysis metadata
            formatted_results.update({
                'processing_time': processing_time,
                'image_path': image_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_shape': image.shape,
                'models_status': self.get_model_status()
            })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            return None
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and validate image using ImageProcessor
        Args:
            image_path: Path to image file
        Returns:
            Image as numpy array or None
        """
        return ImageProcessor.load_image(image_path)
    
    def _calculate_derived_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate additional metrics from analysis results
        Args:
            results: Basic analysis results
        Returns:
            Dictionary with derived metrics
        """
        derived = {}
        
        # Overall confidence score (weighted average)
        classification_confidence = results['classification'].get('confidence', 0)
        segmentation_confidence = 1.0 if results['segmentation'].get('success', False) else 0.5
        
        derived['overall_confidence'] = (classification_confidence * 0.7 + segmentation_confidence * 0.3)
        
        # Risk assessment based on classification
        prediction = results['classification'].get('prediction', 'benign')
        if prediction == 'malignant':
            derived['risk_level'] = 'High'
        else:  # benign
            derived['risk_level'] = 'Medium'
        
        # Tumor presence (from segmentation if available, else from classification)
        if results['segmentation'].get('tumor_detected') is not None:
            derived['tumor_present'] = results['segmentation']['tumor_detected']
        else:
            derived['tumor_present'] = prediction in ['benign', 'malignant']
        
        return derived
    

    

    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of loaded models
        Returns:
            Dictionary with model status information
        """
        return {
            'classification_loaded': self.classification_model.is_loaded(),
            'segmentation_loaded': self.segmentation_model.is_loaded(),
            'classification_info': self.classification_model.get_model_info(),
            'segmentation_info': self.segmentation_model.get_model_info(),
            'backend_ready': self.models_loaded
        }
    
    def load_classification_model(self, model_path: str) -> bool:
        """
        Load or reload classification model
        Args:
            model_path: Path to classification model file
        Returns:
            Success status
        """
        try:
            self.classification_model.model_path = model_path
            success = self.classification_model.load_model()
            self.models_loaded = self.classification_model.is_loaded() or self.segmentation_model.is_loaded()
            return success
        except Exception as e:
            print(f"Error loading classification model: {e}")
            return False
    
    def load_segmentation_model(self, model_path: str) -> bool:
        """
        Load or reload segmentation model
        Args:
            model_path: Path to segmentation model file
        Returns:
            Success status
        """
        try:
            self.segmentation_model.model_path = model_path
            success = self.segmentation_model.load_model()
            self.models_loaded = self.classification_model.is_loaded() or self.segmentation_model.is_loaded()
            return success
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        # Clear models from memory
        self.classification_model.model = None
        self.segmentation_model.model = None
        self.models_loaded = False
        print("ML Backend cleaned up")


class AsyncProcessor:
    """Handle asynchronous processing for responsive UI"""
    
    def __init__(self, ml_backend: MLBackend):
        """
        Initialize async processor
        Args:
            ml_backend: ML backend instance
        """
        self.ml_backend = ml_backend
        self.current_task = None
        self.is_processing = False
        self.progress = 0.0
    
    def analyze_image_async(self, image_path: str, callback):
        """
        Analyze image asynchronously
        Args:
            image_path: Path to image
            callback: Function to call with results
        """
        import threading
        
        if self.is_processing:
            return False
        
        def process():
            try:
                self.is_processing = True
                self.progress = 0.0
                
                # Simulate progress updates
                for i in range(10):
                    time.sleep(0.1)
                    self.progress = (i + 1) / 10.0
                
                # Run actual analysis
                results = self.ml_backend.analyze_image(image_path)
                
                # Call callback with results
                if callback:
                    callback(results)
                
            except Exception as e:
                if callback:
                    callback({'error': str(e)})
            finally:
                self.is_processing = False
                self.progress = 0.0
        
        self.current_task = threading.Thread(target=process)
        self.current_task.daemon = True
        self.current_task.start()
        return True
    
    def cancel_processing(self):
        """Cancel current processing task"""
        if self.current_task and self.current_task.is_alive():
            # Note: Python threading doesn't support clean cancellation
            # This is a placeholder for future implementation
            self.is_processing = False
    
    def get_progress(self) -> float:
        """
        Get current progress
        Returns:
            Progress value between 0.0 and 1.0
        """
        return self.progress


# Example usage and testing
if __name__ == "__main__":
    print("Testing ML Backend...")
    
    # Initialize backend
    backend = MLBackend()
    
    # Test initialization
    success = backend.initialize()
    print(f"Backend initialized: {success}")
    
    # Test model status
    status = backend.get_model_status()
    print(f"Model status: {status}")
    
    # Test analysis with demo image
    print("\\nTesting analysis...")
    results = backend.analyze_image("demo_image.jpg")
    
    print(f"Analysis complete:")
    print(f"- Classification: {results['classification']['prediction']}")
    print(f"- Confidence: {results['classification']['confidence']:.2f}")
    print(f"- Tumor detected: {results['segmentation']['tumor_detected']}")
    print(f"- Processing time: {results['processing_time']:.2f}s")
    print(f"- Risk level: {results['risk_level']}")
    print(f"- Recommendations: {len(results['recommendations'])} items")
    
    # Test async processor
    print("\\nTesting async processor...")
    
    def result_callback(results):
        print(f"Async analysis complete: {results.get('classification', {}).get('prediction', 'Error')}")
    
    async_proc = AsyncProcessor(backend)
    async_proc.analyze_image_async("demo_image.jpg", result_callback)
    
    # Wait a bit for async processing
    time.sleep(2)
    
    print("\\nML Backend testing complete!")