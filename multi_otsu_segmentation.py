import numpy as np
from skimage import io, color, filters
import cv2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def multi_otsu_segmentation(image_path, n_classes=3):
    # Read the image
    image = io.imread(image_path)
    
    # If the image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Apply Multi-Otsu thresholding
    thresholds = filters.threshold_multiotsu(image, classes=n_classes)
    regions = np.digitize(image, bins=thresholds)
    
    # Normalize the segmented image for visualization
    segmented = (regions - regions.min()) / (regions.max() - regions.min())
    
    return segmented, thresholds

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
    segmented_image, thresholds = multi_otsu_segmentation(image_path)
    
    # Save the segmented image
    cv2.imwrite("multi_otsu_segmented.jpg", (segmented_image * 255).astype(np.uint8))
    
    # Print evaluation metrics
    original_image = io.imread(image_path)
    if len(original_image.shape) == 3:
        original_image = color.rgb2gray(original_image)
    
    metrics = evaluate_segmentation(original_image, segmented_image)
    print("Multi-Otsu Segmentation Metrics:")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}")
    print(f"Thresholds: {thresholds}") 