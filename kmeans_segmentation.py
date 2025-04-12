'''import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import cv2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def kmeans_segmentation(image_path, n_clusters=3):
    # Read the image
    image = io.imread(image_path)
    
    # If the image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 1)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 )
    labels = kmeans.fit_predict(pixels)
    
    # Reshape the labels back to the original image shape
    segmented = labels.reshape(image.shape)
    
    # Normalize the segmented image for visualization
    segmented = (segmented - segmented.min()) / (segmented.max() - segmented.min())
    
    return segmented, kmeans

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
    segmented_image, kmeans = kmeans_segmentation(image_path)
    
    # Save the segmented image
    cv2.imwrite("kmeans_segmented.jpg", (segmented_image * 255).astype(np.uint8))
    
    # Print evaluation metrics
    original_image = io.imread(image_path)
    if len(original_image.shape) == 3:
        original_image = color.rgb2gray(original_image)
    
    metrics = evaluate_segmentation(original_image, segmented_image)
    print("K-means Segmentation Metrics:")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}") '''
    
import numpy as np
from sklearn.cluster import KMeans
from skimage import io, color
import cv2
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def kmeans_segmentation(image_path, n_clusters=3):
    # Read the image
    image = io.imread(image_path)
    
    # If the image is RGB, convert to grayscale
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 1)
    
    # Apply K-means clustering with explicit n_init
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')  # <-- updated
    labels = kmeans.fit_predict(pixels)
    
    # Reshape the labels back to the original image shape
    segmented = labels.reshape(image.shape)
    
    # Normalize the segmented image for visualization
    segmented = (segmented - segmented.min()) / (segmented.max() - segmented.min())
    
    return segmented, kmeans

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
    segmented_image, kmeans = kmeans_segmentation(image_path)
    
    # Save the segmented image
    cv2.imwrite("kmeans_segmented.jpg", (segmented_image * 255).astype(np.uint8))
    
    # Print evaluation metrics
    original_image = io.imread(image_path)
    if len(original_image.shape) == 3:
        original_image = color.rgb2gray(original_image)
    
    metrics = evaluate_segmentation(original_image, segmented_image)
    print("K-means Segmentation Metrics:")
    print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}")
    print(f"Normalized Mutual Information: {metrics['normalized_mutual_info']:.4f}")
