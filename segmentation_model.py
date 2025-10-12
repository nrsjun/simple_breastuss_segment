"""
Segmentation Model for Breast Ultrasound Project
U-Net model with ResNet34 encoder for tumor segmentation
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import os
import random

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not available. Install with: pip install segmentation-models-pytorch")


class UNetModel(nn.Module):
    """Basic U-Net architecture for tumor segmentation"""
    
    def __init__(self, input_channels=3, output_channels=1):
        """
        Set up the U-Net model architecture
        Args:
            input_channels: Number of input channels (3 for RGB)
            output_channels: Number of output channels (1 for binary mask)
        """
        super(UNetModel, self).__init__()
        
        # Encoder layers (downsampling path)
        self.encoder1 = self._conv_block(input_channels, 64)
        self.encoder2 = self._conv_block(64, 128)
        self.encoder3 = self._conv_block(128, 256)
        self.encoder4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder layers (upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self._conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self._conv_block(128, 64)
        
        # Final output layer
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def _conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass through the U-Net"""
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        output = self.output(dec1)
        return torch.sigmoid(output)


class SegmentationModel:
    def __init__(self, model_path: str = None):
        """
        Set up the segmentation model
        Args:
            model_path: Path to the trained model file (.pth format)
        """
        self.model = None
        self.model_path = model_path
        self.input_size = (256, 256)  # Model requires 256x256, but accepts any input size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_model_loaded = False
        
    def load_model(self, model_path: str = None) -> bool:
        """
        Load the trained segmentation model
        Args:
            model_path: Path to model file (optional if already set)
            
        Returns:
            bool: True if successful, False if failed
        """
        try:
            # Use provided path or the one set earlier
            path = model_path or self.model_path
            
            if not path or not os.path.exists(path):
                print(f"Cannot find segmentation model file: {path}")
                return False
            
            # Check if segmentation_models_pytorch is available
            if not SMP_AVAILABLE:
                print("No segmenetation model")
                return False
                
            self.model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=1,      # grayscale
                classes=1           # binary mask (logits)
            )
            
            # Load the trained weights
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            self.is_model_loaded = True
            print(f"Segmentation model loaded successfully from: {path}")
            return True
            
        except Exception as e:
            print(f"Error loading segmentation model: {e}")
            print(f"Make sure segmentation_models_pytorch is installed: pip install segmentation-models-pytorch")
            self.is_model_loaded = False
            return False
    
    def preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Prepare image for segmentation model - accepts any size image
        
        Args:
            image: Input image as numpy array (any size)
            
        Returns:
            Processed image ready for segmentation or None if error
        """
        try:
            # Handle different image formats and convert to grayscale (model expects 1 channel)
            if len(image.shape) == 2:
                # Already grayscale
                pass
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Single channel - squeeze the dimension
                image = np.squeeze(image, axis=2)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # BGR/RGB image - convert to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA image - convert to grayscale (ignore alpha)
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            
            # Store original size for later resizing
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Resize from any input size to model's required size (256x256)
            image = cv2.resize(image, self.input_size)
            
            # Normalize pixel values to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            # Convert to PyTorch tensor and add batch and channel dimensions
            # From (H, W) to (1, 1, H, W) - batch_size=1, channels=1
            image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
            image = image.to(self.device)
            
            return image, original_size
            
        except Exception as e:
            print(f"Error processing image for segmentation: {e}")
            return None, None
    
    def postprocess_mask(self, mask: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Post-process the segmentation mask
        
        Args:
            mask: Output mask from model
            original_size: Original image size (width, height)
            
        Returns:
            Processed mask as numpy array
        """
        try:
            # Convert from tensor to numpy
            mask = mask.cpu().numpy()[0, 0]  # Remove batch and channel dimensions
            
            # Resize back to original image size
            mask = cv2.resize(mask, original_size)
            
            # Apply threshold to create binary mask
            mask = (mask > 0.5).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            print(f"Error post-processing mask: {e}")
            return np.zeros(original_size[::-1], dtype=np.uint8)  # Return empty mask
    
    def predict(self, image: np.ndarray) -> Dict[str, any]:
        """
        Segment the ultrasound image to find tumor regions - accepts any size image
        
        Args:
            image: Input ultrasound image as numpy array (any size)
            
        Returns:
            Dictionary with segmentation results
        """
        try:
            # Check if model is ready
            if not self.is_model_loaded or self.model is None:
                return self._get_demo_result(image.shape[:2])
            
            # Process the image first
            processed_image, original_size = self.preprocess_image(image)
            if processed_image is None:
                return self._get_demo_result(image.shape[:2])
            
            # Run segmentation
            with torch.no_grad():
                mask_output = self.model(processed_image)
            
            # Post-process the mask
            mask = self.postprocess_mask(mask_output, original_size)
            
            # Calculate basic metrics
            tumor_detected = np.sum(mask) > 0
            tumor_area = np.sum(mask)
            
            # Create overlay image
            overlay = self._create_overlay(image, mask)
            
            return {
                'mask': mask,
                'tumor_detected': tumor_detected,
                'tumor_area_pixels': int(tumor_area),
                'overlay_image': overlay,
                'success': True
            }
            
        except Exception as e:
            print(f"Error during segmentation: {e}")
            return self._get_demo_result(image.shape[:2])
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create overlay image with mask highlighted"""
        try:
            # Ensure both images are RGB for blending
            if len(image.shape) == 3 and image.shape[2] == 3:
                base_img = image.copy()
                overlay_img = image.copy()
            else:
                # Convert grayscale to RGB for both base and overlay
                base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                overlay_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize mask to match image if needed
            if mask.shape != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Create colored overlay where mask is present
            overlay_img[mask > 0] = [255, 0, 0]  # Red overlay for tumor areas
            
            # Blend with original image (both are now RGB)
            alpha = 0.3
            blended = cv2.addWeighted(base_img, 1-alpha, overlay_img, alpha, 0)
            
            return blended
            
        except Exception as e:
            print(f"Error creating overlay: {e}")
            # Return original image converted to RGB if grayscale
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
    
    def is_loaded(self) -> bool:
        """
        Check if model is loaded and ready
        Returns:
            True if model is loaded, False otherwise
        """
        return self.is_model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the segmentation model
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_loaded': self.is_loaded(),
            'model_path': self.model_path,
            'input_size': self.input_size,
            'device': str(self.device),
            'model_type': 'U-Net'
        }


