from PIL import Image

def get_total_pixels(image_path):
    """Returns the total number of pixels in the given image."""
    img = Image.open(image_path)
    width, height = img.size
    return width * height