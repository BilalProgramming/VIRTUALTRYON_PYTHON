import cv2
import numpy as np
import math
from Utils import overlayPNG

class WristTryOn:
    def __init__(self):
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            print("MediaPipe Hands Initialized")
        except Exception as e:
            print(f"MediaPipe Hands Failed: {e}")
            
        self.watch = cv2.imread("Resources/Watch.png", cv2.IMREAD_UNCHANGED)
        
    def process(self, img):
        if not self.use_mediapipe:
            # Simple UI feedback that MediaPipe is missing for this feature
            cv2.putText(img, "Wrist Tracking requires MediaPipe", (50, 400), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img, "Please install C++ Redistributables", (50, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            return img, []

        import mediapipe as mp
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img.shape
                landmarks_list = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                
                p0 = landmarks_list[0] 
                p9 = landmarks_list[9] 
                p5 = landmarks_list[5]
                p17 = landmarks_list[17]
                palm_width = math.hypot(p5[0]-p17[0], p5[1]-p17[1])
                
                scale = 1.2
                target_width = int(palm_width * scale)
                
                dX, dY = p9[0] - p0[0], p9[1] - p0[1]
                angle = math.degrees(math.atan2(dY, dX)) - 90 
                
                if target_width > 0:
                    try:
                        scale_factor = target_width / self.watch.shape[1]
                        new_h = int(self.watch.shape[0] * scale_factor)
                        resized_watch = cv2.resize(self.watch, (target_width, new_h))
                        
                        center = (target_width // 2, new_h // 2)
                        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        rotated_watch = cv2.warpAffine(resized_watch, M, (target_width, new_h),
                                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                        
                        img = overlayPNG(img, rotated_watch, [int(p0[0] - target_width / 2), int(p0[1] - new_h / 2)])
                    except: pass
                        
        return img, landmarks_list
