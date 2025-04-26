from flask import Flask, render_template, request, send_file, url_for,flash,redirect
import os
from werkzeug.utils import secure_filename
import numpy as np
from kmeans_segmentation import kmeans_segmentation
from multi_otsu_segmentation import multi_otsu_segmentation
from normal_thresholding import normal_thresholding
import cv2
from skimage import io, color
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from original_out import convert_tiff_to_jpg
from totalPixel import get_total_pixels
from PIL import Image
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY') #'super_secret_key_123'  # you can change this to anything random
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def save_image(image, filename, folder):
    """Save image with proper normalization and format"""
    # Ensure the image is in the correct range (0-255)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Create the full path
    full_path = os.path.join(folder, filename)
    
    # Save the image
    cv2.imwrite(full_path, image)
    return filename

def compare_segmentations(original_image, segmentations):
    best_method = None
    best_metrics = {'adjusted_rand_index': -1, 'normalized_mutual_info': -1}
    
    for method_name, segmented_image in segmentations.items():
        # Convert to 1D arrays for evaluation
        original_flat = original_image.reshape(-1)
        segmented_flat = segmented_image.reshape(-1)
        
        # Convert continuous values to discrete labels
        original_labels = np.digitize(original_flat, bins=np.linspace(0, 1, 10))
        segmented_labels = np.digitize(segmented_flat, bins=np.linspace(0, 1, 10))
        
        # Calculate metrics
        ari = adjusted_rand_score(original_labels, segmented_labels)
        nmi = normalized_mutual_info_score(original_labels, segmented_labels)
        
        # Update best method if current metrics are better
        if ari > best_metrics['adjusted_rand_index']:
            best_metrics = {'adjusted_rand_index': ari, 'normalized_mutual_info': nmi}
            best_method = method_name
    
    return best_method, best_metrics

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img = Image.open(filepath)
            width, height = img.size

            if width > 100 or height > 100:
                flash('⚠️ Please upload an image smaller than 100 x 100 pixels.')
                return redirect(url_for('index'))
        except Exception as e:
            flash(f"Error reading image: {e}")
            return redirect(url_for('index'))
        
        Total_p=get_total_pixels(filepath)
        
        if(Total_p>64*64):
            img = Image.open(filepath)
            img = img.resize((64, 64))
            img.save(filepath)
        
        #convert_tiff_to_jpg(filepath, output_path="output\converted_image.jpg")
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            converted_path = convert_tiff_to_jpg(filepath)
            display_image = converted_path.replace("static/", "")  # path relative to static
        else:
            #display_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = cv2.imread(filepath)
            display_filename = 'original_' + filename
            display_path = os.path.join(app.config['STATIC_FOLDER'], display_filename)
            cv2.imwrite(display_path, image)
            display_image = display_filename  # just filename, used with static/

        
        # Read the original image
        original_image = io.imread(filepath)
        if len(original_image.shape) == 3:
            original_image = color.rgb2gray(original_image)
        
        # Normalize the original image
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        
        # Save original image
        original_filename = save_image((original_image * 255).astype(np.uint8), 
                                     'original_' + filename, 
                                     app.config['STATIC_FOLDER'])
        
        #tiff_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        #jpg_image = tiff_image.copy()
        #cv2.imwrite('output_image.jpg', jpg_image)
        
        # Perform all segmentations
        segmentations = {}
        result_filenames = {}
        
        # K-means segmentation
        kmeans_result, _ = kmeans_segmentation(filepath)
        segmentations['kmeans'] = kmeans_result
        result_filenames['kmeans'] = save_image(kmeans_result, 
                                              'kmeans_result.jpg', 
                                              app.config['STATIC_FOLDER'])
        
        # Multi-Otsu segmentation
        multi_otsu_result, _ = multi_otsu_segmentation(filepath)
        segmentations['multi_otsu'] = multi_otsu_result
        result_filenames['multi_otsu'] = save_image(multi_otsu_result, 
                                                  'multi_otsu_result.jpg', 
                                                  app.config['STATIC_FOLDER'])
        
        # Normal thresholding
        normal_threshold_result, _ = normal_thresholding(filepath)
        segmentations['normal_threshold'] = normal_threshold_result
        result_filenames['normal_threshold'] = save_image(normal_threshold_result, 
                                                        'normal_threshold_result.jpg', 
                                                        app.config['STATIC_FOLDER'])
        
        # Compare segmentations
        best_method, best_metrics = compare_segmentations(original_image, segmentations)
        
        # Save the best result
        best_result = segmentations[best_method]
        best_filename = save_image(best_result, 
                                 'best_result.jpg', 
                                 app.config['STATIC_FOLDER'])
        
        # Generate explanation
        explanation = generate_explanation(best_method, best_metrics)
        
        return render_template('results.html',
                             original_image=display_image,
                             #img_show=jpg_image,
                             best_method=best_method,
                             metrics=best_metrics,
                             explanation=explanation,
                             result_filenames=result_filenames)

def generate_explanation(best_method, metrics):
    explanations = {
        'kmeans': "K-means clustering was chosen because it effectively identified distinct clusters in the image data, providing good separation between different regions. This method works well when the image has clear, distinct regions with different intensity values.",
        'multi_otsu': "Multi-Otsu thresholding was selected as it automatically determined optimal thresholds to separate the image into multiple classes. This method is particularly effective when the image histogram shows multiple distinct peaks.",
        'normal_threshold': "Simple thresholding was found to be most effective, suggesting that the image has a clear bimodal distribution where a single threshold can effectively separate the regions of interest."
    }
    
    return explanations.get(best_method, "The segmentation method was chosen based on its superior performance in terms of adjusted rand index and normalized mutual information scores.")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
