# Image Segmentation Comparison Tool

This application compares three different image segmentation methods:
1. K-means Clustering
2. Multi-Otsu Thresholding
3. Normal Thresholding

The system automatically determines which method works best for a given image based on performance metrics.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir static
mkdir uploads
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload an image using the web interface
2. The system will process the image using all three segmentation methods
3. Results will be displayed showing:
   - The best performing method
   - Performance metrics
   - Explanation of why the method was chosen
   - Original and segmented images

## Directory Structure

- `app.py`: Main Flask application
- `kmeans_segmentation.py`: K-means clustering implementation
- `multi_otsu_segmentation.py`: Multi-Otsu thresholding implementation
- `normal_thresholding.py`: Normal thresholding implementation
- `templates/`: HTML templates
  - `index.html`: Upload page
  - `results.html`: Results display page
- `static/`: Directory for storing processed images
- `uploads/`: Directory for storing uploaded images

## Performance Metrics

The system evaluates segmentation quality using:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

Higher values indicate better segmentation performance.

## Requirements

- Python 3.7+
- Flask
- NumPy
- scikit-image
- scikit-learn
- OpenCV
- Matplotlib
- Pillow 