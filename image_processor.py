"""
Image Processing Classes for Breast Ultrasound AI Application
Handles image loading, preprocessing, and result processing
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, Tuple, Optional, Any
import os


class ImageProcessor:
    """Handle all image processing operations"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions
        
        Args:
            image: Input image as numpy array
            target_size: Target size as (width, height)
            
        Returns:
            Resized image
        """
        try:
            return cv2.resize(image, target_size)
        except Exception as e:
            print(f"Error resizing image: {e}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to 0-1 range
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Normalized image
        """
        try:
            return image.astype(np.float32) / 255.0
        except Exception as e:
            print(f"Error normalizing image: {e}")
            return image
    
    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """
        Convert image to RGB format from various input formats
        
        Args:
            image: Input image as numpy array
            
        Returns:
            RGB image
        """
        try:
            # Handle different image formats
            if len(image.shape) == 2:
                # Grayscale image - convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    # Single channel - convert to RGB
                    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 3:
                    # Assume BGR - convert to RGB
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    # RGBA - remove alpha and convert to RGB
                    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            
            return image
            
        except Exception as e:
            print(f"Error converting to RGB: {e}")
            return image
    
    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if image is properly loaded and formatted
        
        Args:
            image: Image to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if image is None:
                return False
            
            if not isinstance(image, np.ndarray):
                return False
            
            if len(image.shape) < 2 or len(image.shape) > 3:
                return False
            
            if image.size == 0:
                return False
            
            return True
            
        except Exception:
            return False


class ResultProcessor:
    """Process and enhance analysis results"""
    
    @staticmethod
    def calculate_tumor_metrics(mask: np.ndarray) -> Dict[str, float]:
        """
        Calculate tumor measurements from segmentation mask - pixels only
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            Dictionary with tumor pixel count
        """
        try:
            if mask is None or np.sum(mask) == 0:
                return {'area_pixels': 0.0}
            
            # Just count the pixels
            area = np.sum(mask > 0)
            
            return {'area_pixels': float(area)}
            
        except Exception as e:
            print(f"Error calculating tumor metrics: {e}")
            return {'area_pixels': 0.0}
    
    @staticmethod
    def detect_tumor_presence(mask: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Detect if tumor is present based on segmentation mask
        
        Args:
            mask: Segmentation mask
            threshold: Minimum area threshold (as fraction of image)
            
        Returns:
            True if tumor detected, False otherwise
        """
        try:
            if mask is None:
                return False
            
            # Calculate tumor area as fraction of total image
            total_pixels = mask.shape[0] * mask.shape[1]
            tumor_pixels = np.sum(mask > 0)
            tumor_fraction = tumor_pixels / total_pixels
            
            return tumor_fraction > (threshold / 100.0)  # Convert threshold to fraction
            
        except Exception as e:
            print(f"Error detecting tumor presence: {e}")
            return False
    
    @staticmethod
    def generate_overlay(image: np.ndarray, mask: np.ndarray, 
                        color: Tuple[int, int, int] = (255, 0, 0),
                        alpha: float = 0.3) -> np.ndarray:
        """
        Generate overlay image with mask highlighted
        
        Args:
            image: Original image
            mask: Segmentation mask
            color: Overlay color (R, G, B)
            alpha: Overlay transparency (0-1)
            
        Returns:
            Image with overlay
        """
        try:
            if mask is None or image is None:
                return image
            
            # Ensure image is RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                overlay_img = image.copy()
            else:
                overlay_img = ImageProcessor.convert_to_rgb(image)
            
            # Resize mask to match image if needed
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create colored overlay
            colored_mask = np.zeros_like(overlay_img)
            colored_mask[mask > 0] = color
            
            # Blend with original image
            blended = cv2.addWeighted(overlay_img, 1-alpha, colored_mask, alpha, 0)
            
            return blended
            
        except Exception as e:
            print(f"Error generating overlay: {e}")
            return image
    
    @staticmethod
    def format_results(classification: Dict[str, Any], 
                      segmentation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format and combine analysis results
        
        Args:
            classification: Classification results
            segmentation: Segmentation results
            
        Returns:
            Formatted combined results
        """
        try:
            # Calculate overall confidence
            cls_confidence = classification.get('confidence', 0.0)
            seg_success = segmentation.get('success', False)
            seg_confidence = 0.8 if seg_success else 0.5
            
            overall_confidence = (cls_confidence * 0.7 + seg_confidence * 0.3)
            
            # Determine risk level
            prediction = classification.get('prediction', 'benign')
            if prediction == 'malignant':
                risk_level = 'High'
            else:  # benign
                risk_level = 'Medium'
            
            # Generate summary
            tumor_detected = segmentation.get('tumor_detected', False)
            tumor_area = segmentation.get('tumor_area_pixels', 0)
            
            summary = {
                'prediction': prediction,
                'confidence': cls_confidence,
                'overall_confidence': overall_confidence,
                'risk_level': risk_level,
                'tumor_detected': tumor_detected,
                'tumor_area': tumor_area,
                'requires_followup': prediction in ['benign', 'malignant'] or tumor_detected
            }
            
            return {
                'classification': classification,
                'segmentation': segmentation,
                'summary': summary
            }
            
        except Exception as e:
            print(f"Error formatting results: {e}")
            return {
                'classification': classification,
                'segmentation': segmentation,
                'summary': {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'risk_level': 'Unknown',
                    'tumor_detected': False
                }
            }
    



class FileManager:
    """Handle file operations for saving reports"""
    
    @staticmethod
    def save_analysis_report(results: Dict[str, Any], output_path: str) -> bool:
        """
        Save analysis results to a text file
        Args:
            results: Analysis results dictionary
            output_path: Output file path
        Returns:
            Success status
        """
        try:
            from datetime import datetime
            
            lines = []
            lines.append("BREAST ULTRASOUND AI ANALYSIS REPORT")
            lines.append("=" * 50)
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Classification results
            if 'classification' in results:
                cls_result = results['classification']
                lines.append("CLASSIFICATION RESULTS:")
                lines.append(f"Prediction: {cls_result.get('prediction', 'Unknown').upper()}")
                lines.append(f"Confidence: {cls_result.get('confidence', 0)*100:.1f}%")
                lines.append("")
                
                probabilities = cls_result.get('probabilities', {})
                if probabilities:
                    lines.append("Class Probabilities:")
                    for class_name, prob in probabilities.items():
                        lines.append(f"  {class_name.title()}: {prob*100:.1f}%")
                    lines.append("")
            
            # Segmentation results
            if 'segmentation' in results:
                seg_result = results['segmentation']
                lines.append("SEGMENTATION RESULTS:")
                lines.append(f"Tumor Detected: {'Yes' if seg_result.get('tumor_detected', False) else 'No'}")
                if seg_result.get('tumor_detected', False):
                    lines.append(f"Tumor Area: {seg_result.get('tumor_area_pixels', 0)} pixels")
                lines.append("")
            
            # Summary
            if 'summary' in results:
                summary = results['summary']
                lines.append("ANALYSIS SUMMARY:")
                lines.append(f"Risk Level: {summary.get('risk_level', 'Unknown')}")
                lines.append(f"Overall Confidence: {summary.get('overall_confidence', 0)*100:.1f}%")
                lines.append("")
            
            # Processing info
            processing_time = results.get('processing_time', 0)
            timestamp = results.get('timestamp', 'Unknown')
            lines.append("PROCESSING INFO:")
            lines.append(f"Processing Time: {processing_time:.2f} seconds")
            lines.append(f"Analysis Date: {timestamp}")
            lines.append(f"Image Path: {results.get('image_path', 'Unknown')}")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\\n'.join(lines))
            
            return True
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
