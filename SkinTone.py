import cv2
import numpy as np

def analyze_skin_tone(img, landmarks):
    """
    Analyzes skin tone from face landmarks (cheek area) and recommends Gold or Silver.
    Returns: 'Gold' (Warm) or 'Silver' (Cool) and the color.
    """
    height, width, _ = img.shape
    
    # Cheek landmarks (approximate indices for left cheek)
    # 50, 117, 123, 147 
    # Let's take a small ROI around landmark 50 (cheek)
    cheek_idx = 50
    if cheek_idx >= len(landmarks):
        return "Detecting...", (128, 128, 128)
        
    point = landmarks[cheek_idx]
    cx, cy = int(point[0]), int(point[1])
    
    # Define ROI
    size = 10
    x1, y1 = max(0, cx - size), max(0, cy - size)
    x2, y2 = min(width, cx + size), min(height, cy + size)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return "Detecting...", (128, 128, 128)
        
    # Calculate average color in BGR
    avg_color_row = np.average(roi, axis=0)
    avg_color = np.average(avg_color_row, axis=0)
    b, g, r = avg_color
    
    # Convert BGR to YCrCb (better for skin detection, but for warm/cool we can stick to Lab or simple heuristics)
    # Simple Heuristic:
    # Warm (Gold): Higher R and G relative to B
    # Cool (Silver): Higher B relative to others, or less contrast between R and B
    
    # Using CIELAB b* channel is better for warm/cool
    # +b is yellow (Warm), -b is blue (Cool)
    
    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    avg_lab_row = np.average(roi_lab, axis=0)
    avg_lab = np.average(avg_lab_row, axis=0)
    L, a, b_star = avg_lab
    
    # Threshold for b_star (Yellow-Blue axis)
    # Higher b* means more yellow/warm.
    
    recommendation = "Silver"
    if b_star > 145: # Typical skin adjustment, standard neutral is ~128 in 8-bit Lab but OpenCV mapping is different
        # OpenCV Lat: L (0-255), a (0-255), b (0-255). Neutral is 128.
        # > 128 is Warm (Yellow), < 128 is Cool (Blue)
        # However skin is generally warm. We need a relative threshold.
        # Let's say if it's VERY warm (>150) -> Gold.
        # If it's neutral/pinkish (<145) -> Silver.
        recommendation = "Gold"
    else:
        recommendation = "Silver"
        
    return recommendation, (int(b), int(g), int(r))
