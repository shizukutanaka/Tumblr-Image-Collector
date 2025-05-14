import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

class ImageClassifier:
    def __init__(self, nsfw_threshold=0.7, min_resolution=(300, 300)):
        """
        Initialize the image classifier with pre-trained models and configuration.
        
        :param nsfw_threshold: Threshold for detecting potentially inappropriate content
        :param min_resolution: Minimum acceptable image resolution
        """
        # Load pre-trained MobileNetV2 model for general image classification
        self.classification_model = MobileNetV2(weights='imagenet')
        
        # Configuration parameters
        self.nsfw_threshold = nsfw_threshold
        self.min_resolution = min_resolution
    
    def analyze_image(self, image_path):
        """
        Perform comprehensive image analysis.
        
        :param image_path: Path to the image file
        :return: Dictionary with image analysis results
        """
        try:
            # Open image and perform initial checks
            with Image.open(image_path) as img:
                # Check resolution
                width, height = img.size
                is_high_resolution = (width >= self.min_resolution[0] and 
                                      height >= self.min_resolution[1])
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Prepare image for classification
                img_array = img_to_array(img.resize((224, 224)))
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Predict image contents
                predictions = self.classification_model.predict(img_array)
                decoded_predictions = decode_predictions(predictions, top=3)[0]
                
                # Basic NSFW detection (placeholder - you'd want a more sophisticated model)
                # This is a very basic approximation and should be replaced with a dedicated NSFW model
                nsfw_keywords = ['nude', 'lingerie', 'bikini', 'underwear']
                is_potentially_nsfw = any(
                    any(keyword in pred[1].lower() for keyword in nsfw_keywords) 
                    for pred in decoded_predictions
                )
                
                return {
                    'is_valid': is_high_resolution and not is_potentially_nsfw,
                    'resolution': (width, height),
                    'is_high_resolution': is_high_resolution,
                    'is_potentially_nsfw': is_potentially_nsfw,
                    'top_predictions': [
                        {'label': pred[1], 'confidence': float(pred[2])} 
                        for pred in decoded_predictions
                    ]
                }
        
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e)
            }

def classify_tumblr_images(image_paths, classifier=None):
    """
    Utility function to classify multiple images.
    
    :param image_paths: List of image file paths
    :param classifier: Optional ImageClassifier instance
    :return: Dictionary of classification results
    """
    if classifier is None:
        classifier = ImageClassifier()
    
    results = {}
    for image_path in image_paths:
        results[image_path] = classifier.analyze_image(image_path)
    
    return results

# Example usage
if __name__ == '__main__':
    # Example of how to use the classifier
    test_images = [
        'path/to/image1.jpg',
        'path/to/image2.png'
    ]
    
    classification_results = classify_tumblr_images(test_images)
    for image_path, result in classification_results.items():
        print(f"Image: {image_path}")
        print(f"Analysis: {result}")
