import cv2
import os
import numpy as np

def convert_tiff_to_jpg(input_path, output_path="output/converted_image.jpg"):
    # Load the TIFF image
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    # Safely get the output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the image as JPEG
    cv2.imwrite(output_path, image)

    return output_path
