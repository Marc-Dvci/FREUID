import cv2
import numpy as np

def extract_features(img_path):
    """
    Extracts handcrafted forensic features from an image.
    Useful as a baseline when we don't have enough data to train deep models.
    """
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(5)
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge sharpness / Laplacian variance (Blur detection)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Noise estimation
    # Fast estimation of noise level
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.abs(gray.astype(float) - blur.astype(float))
    noise_mean = np.mean(noise)
    noise_std = np.std(noise)
    
    # 3. High-frequency energy (can indicate print/scan artifacts or moire)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    # Mask out the low frequencies (center of spectrum)
    radius = min(h, w) // 4
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    high_freq_mask = dist_from_center > radius
    high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
    
    # 4. Color variance (fraudulent docs often have poor color reproduction)
    color_std = np.std(img, axis=(0, 1)).mean()
    
    return np.array([
        laplacian_var,
        noise_mean,
        noise_std,
        high_freq_energy,
        color_std
    ])
