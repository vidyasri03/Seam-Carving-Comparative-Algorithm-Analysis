import cv2
import numpy as np

# Sobel energy map calculation
def compute_energy_map_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = cv2.magnitude(grad_x, grad_y)
    return energy_map

# Laplacian energy map calculation
def compute_energy_map_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    energy_map = np.abs(laplacian)
    return energy_map

# Function to apply red colormap
def apply_red_colormap(energy_map):
    energy_map_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX)
    energy_map_normalized = np.uint8(energy_map_normalized)
    red_map = cv2.merge([energy_map_normalized, np.zeros_like(energy_map_normalized), np.zeros_like(energy_map_normalized)])
    return red_map
