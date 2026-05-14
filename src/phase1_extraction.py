import cv2
import numpy as np
import math

def extract_stroke_thickness(image_path):
    """
    Calculates average stroke thickness τ = (1/N) * Σ(Ai/Pi)
    where Ai is area, Pi is perimeter, and area > 10 pixels.
    Implements Eq. 1 from the paper.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    thicknesses = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10:  # Paper specification
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                thicknesses.append(area / perimeter)
                
    if not thicknesses:
        return 1.0 # Default fallback
        
    return sum(thicknesses) / len(thicknesses)

def extract_slant_angle(image_path):
    """
    Calculates slant angle θ using Canny edges and Hough Line Transform.
    Implements Eq. 2 from the paper.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    
    # minLineLength and maxLineGap can be tuned, using standard defaults
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
    
    if lines is None:
        return 0.0 # Default fallback if no lines detected
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # θ = arctan2(y2 - y1, x2 - x1) * 180 / π  (Eq. 2)
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)
        angles.append(angle_deg)
        
    return sum(angles) / len(angles)

def get_style_vector(image_path):
    """
    Returns the style vector s = [τ, θ]
    Implements Eq. 3: s = Φ(x) = [τ, θ] ∈ ℝᵈ
    """
    tau = extract_stroke_thickness(image_path)
    theta = extract_slant_angle(image_path)
    return np.array([tau, theta])

# Quick local test
if __name__ == "__main__":
    # Test on a REAL messy image from your dataset
    test_image = "../data/test/sample.png" 
    
    print(f"Testing OpenCV extraction on: {test_image}")
    s = get_style_vector(test_image)
    print(f"Extracted Style Vector [Thickness, Angle]: {s}")
