"""
Data Management Classes for Breast Ultrasound AI Application
Handles patient data, analysis results, and file operations
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import cv2
import numpy as np


class DataManager:
    """Manage patient data and analysis results"""
    
    def __init__(self, data_dir: str = "patient_data"):
        """
        Initialize data manager
        Args:
            data_dir: Directory to store patient data
        """
        self.data_dir = data_dir
        self.patients_file = os.path.join(data_dir, "patients.json")
        self.analyses_dir = os.path.join(data_dir, "analyses")
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.analyses_dir, exist_ok=True)
        
        # Initialize patients file if needed
        if not os.path.exists(self.patients_file):
            self._save_patients_data({})
    
    def _load_patients_data(self) -> Dict[str, Any]:
        """Load patients data from JSON file"""
        try:
            with open(self.patients_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_patients_data(self, data: Dict[str, Any]) -> None:
        """Save patients data to JSON file"""
        try:
            with open(self.patients_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Failed to save patient data: {e}")
    
    def create_patient(self, patient_data: Dict[str, Any]) -> str:
        """
        Create new patient record
        Args:
            patient_data: Patient information dictionary
        Returns:
            Patient ID
        """
        patients = self._load_patients_data()
        
        # Generate patient ID if not provided
        patient_id = patient_data.get('patient_id')
        if not patient_id:
            patient_id = f"P{len(patients) + 1:04d}"
            patient_data['patient_id'] = patient_id
        
        # Add timestamps
        patient_data['created_date'] = datetime.now().isoformat()
        patient_data['last_updated'] = datetime.now().isoformat()
        patient_data['analyses'] = []
        
        # Save patient
        patients[patient_id] = patient_data
        self._save_patients_data(patients)
        
        return patient_id
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Get patient by ID
        Args:
            patient_id: Patient identifier
        Returns:
            Patient data or None
        """
        patients = self._load_patients_data()
        return patients.get(patient_id)
    
    def update_patient(self, patient_id: str, patient_data: Dict[str, Any]) -> bool:
        """
        Update patient record
        Args:
            patient_id: Patient identifier
            patient_data: Updated patient data
        Returns:
            Success status
        """
        try:
            patients = self._load_patients_data()
            
            if patient_id not in patients:
                return False
            
            # Preserve some fields
            patient_data['patient_id'] = patient_id
            patient_data['created_date'] = patients[patient_id].get('created_date')
            patient_data['analyses'] = patients[patient_id].get('analyses', [])
            patient_data['last_updated'] = datetime.now().isoformat()
            
            patients[patient_id] = patient_data
            self._save_patients_data(patients)
            return True
            
        except Exception:
            return False
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Delete patient record
        Args:
            patient_id: Patient identifier
        Returns:
            Success status
        """
        try:
            patients = self._load_patients_data()
            
            if patient_id in patients:
                del patients[patient_id]
                self._save_patients_data(patients)
                
                # Delete analysis files
                analysis_file = os.path.join(self.analyses_dir, f"{patient_id}_analyses.json")
                if os.path.exists(analysis_file):
                    os.remove(analysis_file)
                
                return True
            return False
            
        except Exception:
            return False
    
    def list_patients(self) -> List[Dict[str, Any]]:
        """
        Get list of all patients
        Returns:
            List of patient records
        """
        patients = self._load_patients_data()
        return list(patients.values())
    
    def save_analysis(self, patient_id: str, analysis_data: Dict[str, Any]) -> bool:
        """
        Save analysis results for patient
        Args:
            patient_id: Patient identifier
            analysis_data: Analysis results
        Returns:
            Success status
        """
        try:
            # Add metadata
            analysis_data['analysis_id'] = f"A{int(datetime.now().timestamp())}"
            analysis_data['timestamp'] = datetime.now().isoformat()
            analysis_data['patient_id'] = patient_id
            
            # Load existing analyses
            analysis_file = os.path.join(self.analyses_dir, f"{patient_id}_analyses.json")
            analyses = []
            
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analyses = json.load(f)
            
            # Add new analysis
            analyses.append(analysis_data)
            
            # Save analyses
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False)
            
            # Update patient record
            patients = self._load_patients_data()
            if patient_id in patients:
                if 'analyses' not in patients[patient_id]:
                    patients[patient_id]['analyses'] = []
                patients[patient_id]['analyses'].append(analysis_data['analysis_id'])
                patients[patient_id]['last_updated'] = datetime.now().isoformat()
                self._save_patients_data(patients)
            
            return True
            
        except Exception as e:
            print(f"Error saving analysis: {e}")
            return False
    
    def get_patient_analyses(self, patient_id: str) -> List[Dict[str, Any]]:
        """
        Get all analyses for a patient
        Args:
            patient_id: Patient identifier
        Returns:
            List of analysis records
        """
        try:
            analysis_file = os.path.join(self.analyses_dir, f"{patient_id}_analyses.json")
            
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
            
        except Exception:
            return []


class FileManager:
    """Handle file operations and validation"""
    
    @staticmethod
    def validate_image_file(file_path: str) -> bool:
        """
        Validate if file is a supported image
        Args:
            file_path: Path to file
        Returns:
            True if valid image file
        """
        try:
            # Check file extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext not in valid_extensions:
                return False
            
            # Try to load image
            image = cv2.imread(file_path)
            return image is not None
            
        except Exception:
            return False
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """
        Get list of supported image formats
        Returns:
            List of supported file extensions
        """
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    @staticmethod
    def save_analysis_report(results: Dict[str, Any], output_path: str) -> bool:
        """
        Save analysis results to file
        Args:
            results: Analysis results dictionary
            output_path: Output file path
        Returns:
            Success status
        """
        try:
            # Create report text
            lines = []
            lines.append("BREAST ULTRASOUND AI ANALYSIS REPORT")
            lines.append("=" * 50)
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("")
            
            # Classification results
            if 'classification' in results:
                cls_result = results['classification']
                lines.append("CLASSIFICATION RESULTS:")
                lines.append(f"Prediction: {cls_result['prediction'].upper()}")
                lines.append(f"Confidence: {cls_result['confidence']*100:.1f}%")
                lines.append("")
                lines.append("Class Probabilities:")
                for class_name, prob in cls_result['probabilities'].items():
                    lines.append(f"  {class_name.title()}: {prob*100:.1f}%")
                lines.append("")
            
            # Segmentation results
            if 'segmentation' in results:
                seg_result = results['segmentation']
                lines.append("SEGMENTATION RESULTS:")
                lines.append(f"Tumor Detected: {'Yes' if seg_result['tumor_detected'] else 'No'}")
                if seg_result['tumor_detected']:
                    lines.append(f"Tumor Area: {seg_result['tumor_area_pixels']} pixels")
                lines.append("")
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\\n'.join(lines))
            
            return True
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
    
    @staticmethod
    def create_output_directory(base_path: str) -> str:
        """
        Create output directory with timestamp
        Args:
            base_path: Base directory path
        Returns:
            Created directory path
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_path, f"analysis_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


class ConfigManager:
    """Manage application configuration"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize config manager
        Args:
            config_path: Path to config file
        """
        self.config_path = config_path
        self.default_config = {
            'models': {
                'classification_model_path': '',
                'segmentation_model_path': '',
                'auto_load_models': True
            },
            'ui': {
                'window_width': 1400,
                'window_height': 900,
                'theme': 'clam'
            },
            'data': {
                'patient_data_dir': 'patient_data',
                'auto_save_results': True
            },
            'analysis': {
                'segmentation_threshold': 0.5,
                'overlay_alpha': 0.3
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Merge with defaults (in case new keys were added)
                merged_config = self.default_config.copy()
                self._deep_update(merged_config, config)
                return merged_config
            else:
                return self.default_config.copy()
                
        except Exception:
            return self.default_config.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file
        Args:
            config: Configuration dictionary
        Returns:
            Success status
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    def get_model_paths(self) -> Dict[str, str]:
        """
        Get model file paths from config
        Returns:
            Dictionary with model paths
        """
        config = self.load_config()
        return config.get('models', {})
    
    def get_ui_settings(self) -> Dict[str, Any]:
        """
        Get UI settings from config
        Returns:
            Dictionary with UI settings
        """
        config = self.load_config()
        return config.get('ui', {})
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

