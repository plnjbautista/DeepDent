import os
import cv2
import numpy as np
from scipy import special

# K-CFDO Enhancement Function
def k_cfdo_enhancement(image, rho=0.5, k=1.5):
    """
    Enhance the input image using the K-CFDO algorithm.
    
    Parameters:
    -----------
    image : np.ndarray
        Input grayscale image to be enhanced.
    rho : float, optional
        Regularization parameter (default is 0.5).
    k : float, optional
        Scaling factor for kernel filter strength (default is 1.5).
        
    Returns:
    --------
    np.ndarray
        Enhanced image.
    """
    img = image.astype(np.float64)
    if img.max() > 1.0:
        img = img / 255.0  # Normalize if the image is in 8-bit format
    
    n, m = img.shape
    enhanced_img = np.zeros_like(img)
    
    for i in range(n):
        for j in range(m):
            r = img[i, j]
            numerator = r ** ((1 - rho) / k)
            denominator = special.gamma(2 - rho / k)
            enhanced_img[i, j] = img[i, j] * (numerator / denominator)
    
    enhanced_img = (enhanced_img - enhanced_img.min()) / (enhanced_img.max() - enhanced_img.min())
    return (enhanced_img * 255).astype(np.uint8)

def enhance_image(input_image_path, output_image_path, rho=0.5, k=1.5):
    """
    Enhance a single image using the K-CFDO enhancement function and save it to the specified path.
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input image to be enhanced.
    output_image_path : str
        Path where the enhanced image will be saved.
    rho : float, optional
        Regularization parameter (default is 0.5).
    k : float, optional
        Scaling factor for kernel filter strength (default is 1.5).
    """
    # Read the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply enhancement
    enhanced_img = k_cfdo_enhancement(img, rho=rho, k=k)
    
    # Save the enhanced image
    cv2.imwrite(output_image_path, enhanced_img)
    print(f"Enhanced image saved to {output_image_path}")

def enhance_image_from_user(input_image_path, output_dir, rho=0.5, k=1.5):
    """
    Enhances an image from user input and saves it in the specified directory.
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input image to be enhanced.
    output_dir : str
        Directory where the enhanced image will be saved.
    rho : float, optional
        Regularization parameter (default is 0.5).
    k : float, optional
        Scaling factor for kernel filter strength (default is 1.5).
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the output file path based on input image name
    filename = os.path.basename(input_image_path)
    output_image_path = os.path.join(output_dir, filename)
    
    # Enhance the image
    enhance_image(input_image_path, output_image_path, rho, k)
