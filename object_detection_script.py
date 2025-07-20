import cv2
import numpy as np
import os
from ultralytics import YOLO
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json

class TShirtDetector:
    def __init__(self, model_path=None):
        """
        Initialize the T-shirt detector using YOLO model
        
        Args:
            model_path (str): Path to custom YOLO model or None for default
        """
        # Load YOLO model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Use pre-trained YOLOv8 model
            self.model = YOLO('yolov8n.pt')
        
        # Categories that might contain t-shirts/clothing
        self.clothing_categories = [
            'person',  # Often contains clothing
            'shirt',   # If available in custom model
            'clothing', # If available in custom model
            'top',     # If available in custom model
            'tshirt'   # If available in custom model
        ]
        
        # Create directories for output
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for output"""
        self.output_dirs = {
            'cropped_tshirts': 'cropped_tshirts',
            'detection_results': 'detection_results',
            'processed_images': 'processed_images'
        }
        
        for dirname in self.output_dirs.values():
            os.makedirs(dirname, exist_ok=True)
    
    def detect_objects(self, image_path):
        """
        Detect objects in an image using YOLO
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            results: YOLO detection results
        """
        results = self.model(image_path)
        return results
    
    def is_tshirt_like(self, class_name, confidence=0.5):
        """
        Determine if detected object is likely a t-shirt
        
        Args:
            class_name (str): Name of detected class
            confidence (float): Confidence threshold
            
        Returns:
            bool: True if likely a t-shirt
        """
        # This is a simplified approach - in reality, you'd need a model trained
        # specifically for clothing detection or use a clothing-specific dataset
        tshirt_keywords = ['shirt', 'top', 'clothing', 'tshirt', 'blouse', 'jersey']
        
        for keyword in tshirt_keywords:
            if keyword.lower() in class_name.lower():
                return True
        
        # If detecting person, we assume they're wearing clothing
        if 'person' in class_name.lower():
            return True
            
        return False
    
    def crop_tshirt_region(self, image, bbox, padding=20):
        """
        Crop t-shirt region from image with some padding
        
        Args:
            image (numpy.ndarray): Input image
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
            padding (int): Padding around the bounding box
            
        Returns:
            numpy.ndarray: Cropped image
        """
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]
        
        # Add padding and ensure coordinates are within image bounds
        x1 = max(0, int(x1 - padding))
        y1 = max(0, int(y1 - padding))
        x2 = min(width, int(x2 + padding))
        y2 = min(height, int(y2 + padding))
        
        # Crop the region
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def process_image(self, image_path, save_detections=True):
        """
        Process a single image to detect and crop t-shirts
        
        Args:
            image_path (str): Path to input image
            save_detections (bool): Whether to save detection visualization
            
        Returns:
            list: List of cropped t-shirt images
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return []
        
        # Get detections
        results = self.detect_objects(image_path)
        
        cropped_tshirts = []
        detection_info = []
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class name and confidence
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf)
                    
                    # Check if this could be a t-shirt
                    if self.is_tshirt_like(class_name, confidence) and confidence > 0.3:
                        # Get bounding box coordinates
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        # Crop the t-shirt region
                        cropped = self.crop_tshirt_region(image, bbox)
                        
                        if cropped.size > 0:
                            cropped_tshirts.append(cropped)
                            detection_info.append({
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': bbox.tolist()
                            })
        
        # Save detection visualization if requested
        if save_detections and detection_info:
            self.save_detection_visualization(image_path, image, detection_info)
        
        return cropped_tshirts, detection_info
    
    def save_detection_visualization(self, original_path, image, detection_info):
        """
        Save image with detection bounding boxes
        
        Args:
            original_path (str): Path to original image
            image (numpy.ndarray): Image array
            detection_info (list): List of detection information
        """
        # Draw bounding boxes on image
        vis_image = image.copy()
        
        for info in detection_info:
            bbox = info['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{info['class_name']}: {info['confidence']:.2f}"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        filename = os.path.basename(original_path)
        save_path = os.path.join(self.output_dirs['detection_results'], 
                                f'detection_{filename}')
        cv2.imwrite(save_path, vis_image)
    
    def process_dataset(self, image_folder, csv_file=None):
        """
        Process entire dataset of images
        
        Args:
            image_folder (str): Path to folder containing images
            csv_file (str): Optional CSV file with image metadata
            
        Returns:
            dict: Processing results
        """
        results = {
            'processed_images': 0,
            'total_detections': 0,
            'cropped_tshirts': 0,
            'failed_images': []
        }
        
        # Get list of image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(image_folder) 
                              if f.lower().endswith(ext)])
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {image_file}")
            
            image_path = os.path.join(image_folder, image_file)
            
            try:
                cropped_tshirts, detection_info = self.process_image(image_path)
                
                # Save cropped t-shirts
                for j, cropped in enumerate(cropped_tshirts):
                    filename = f"tshirt_{i}_{j}.jpg"
                    save_path = os.path.join(self.output_dirs['cropped_tshirts'], 
                                           filename)
                    cv2.imwrite(save_path, cropped)
                
                results['processed_images'] += 1
                results['total_detections'] += len(detection_info)
                results['cropped_tshirts'] += len(cropped_tshirts)
                
                # Save detection metadata
                metadata = {
                    'original_image': image_file,
                    'detections': detection_info,
                    'cropped_count': len(cropped_tshirts)
                }
                
                metadata_path = os.path.join(self.output_dirs['processed_images'], 
                                           f"metadata_{i}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                results['failed_images'].append(image_file)
        
        return results
    
    def extract_features(self, cropped_image):
        """
        Extract features from cropped t-shirt image for further processing
        
        Args:
            cropped_image (numpy.ndarray): Cropped t-shirt image
            
        Returns:
            dict: Extracted features
        """
        # Convert to PIL Image for feature extraction
        pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        
        # Basic features
        height, width = cropped_image.shape[:2]
        
        # Color features
        mean_color = np.mean(cropped_image.reshape(-1, 3), axis=0)
        
        # Texture features (simplified)
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'dimensions': [width, height],
            'aspect_ratio': width / height,
            'mean_color': mean_color.tolist(),
            'area': width * height,
            'brightness': np.mean(gray)
        }
        
        return features

def main():
    """Main function to run t-shirt detection"""
    # Initialize detector
    detector = TShirtDetector()
    
    # Process images from the downloaded folder
    image_folder = "Amazon Images"
    
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' not found!")
        print("Please run image_download_script.py first to download images.")
        return
    
    print("Starting t-shirt detection and cropping...")
    
    # Process the dataset
    results = detector.process_dataset(image_folder, "real_fashion_data.csv")
    
    # Print results
    print("\n" + "="*50)
    print("PROCESSING RESULTS")
    print("="*50)
    print(f"Processed images: {results['processed_images']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"Cropped t-shirts: {results['cropped_tshirts']}")
    print(f"Failed images: {len(results['failed_images'])}")
    
    if results['failed_images']:
        print("\nFailed images:")
        for img in results['failed_images']:
            print(f"  - {img}")
    
    print(f"\nCropped t-shirts saved to: {detector.output_dirs['cropped_tshirts']}")
    print(f"Detection results saved to: {detector.output_dirs['detection_results']}")
    print(f"Metadata saved to: {detector.output_dirs['processed_images']}")

if __name__ == "__main__":
    main()
