import cv2
import numpy as np

def analyze_skin_tone(img, landmarks):
    """
    Analyzes skin tone from face landmarks (cheek area) and recommends Gold or Silver.
    Returns: Recommendation, Label, and the BGR color.
    """
    height, width, _ = img.shape
    
    # Cheek landmarks (approximate indices for left cheek)
    cheek_idx = 50
    if cheek_idx >= len(landmarks):
        return "Silver", "Unknown", (128, 128, 128)
        
    point = landmarks[cheek_idx]
    cx, cy = int(point[0]), int(point[1])
    
    # Define ROI
    size = 15
    x1, y1 = max(0, cx - size), max(0, cy - size)
    x2, y2 = min(width, cx + size), min(height, cy + size)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return "Silver", "Unknown", (128, 128, 128)
        
    # Calculate average color in BGR
    avg_color = np.mean(roi, axis=(0, 1))
    b, g, r = avg_color
    
    # Convert ROI to LAB for better warm/cool detection
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    avg_lab = np.mean(roi_lab, axis=(0, 1))
    L, a_star, b_star = avg_lab
    
    # Neutral is around 128 in OpenCV's LAB mapping
    # b* > 128 is yellow (Warm), b* < 128 is blue (Cool)
    
    status = "Neutral"
    recommendation = "Silver"
    
    if b_star > 142:
        recommendation = "Gold"
        status = "Warm"
    elif b_star < 135:
        recommendation = "Silver"
        status = "Cool"
    else:
        recommendation = "Silver"
        status = "Neutral"
        
    return recommendation, status, (int(b), int(g), int(r))
