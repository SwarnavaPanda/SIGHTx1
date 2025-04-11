import numpy as np
from skimage import io, color
import cv2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def normal_thresholding(image_path, threshold=0.5):
    # Read the image
    image = io.imread(image_path)
    
    # If the image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Normalize the image
    image = (image - image.min()) / (image.max() - image.min())
    
    # Apply thresholding
    segmented = np.zeros_like(image)
    segmented[image > threshold] = 1
    
    return segmented, threshold

def evaluate_segmentation(original, segmented):
    # Convert to 1D arrays for evaluation
    original_flat = original.reshape(-1)
    segmented_flat = segmented.reshape(-1)
    
    # Calculate metrics
    ari = adjusted_rand_score(original_flat, segmented_flat)
    nmi = normalized_mutual_info_score(original_flat, segmented_flat)
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }

if __name__ == "__main__":
    # Example usage
    image_path = "test_image.jpg"  # Replace with your image path
    segmented_image, threshold = normal_thresholding(image_path)
    
    # Save the segmented image
    cv2.imwrite("normal_thresholded.jpg", (segmented_image * 255).astype(np.uint8))
    
    # Print evaluation metrics
    original_image = io.imread(image_path)
    if len(original_image.shape) == 3:
        original_image = color.rgb2gray(original_image)
    
    metrics = evaluate_segmentation(original_image, segmented_image)
    print("Normal Thresholding Metrics:")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}")
    print(f"Threshold used: {threshold}") 