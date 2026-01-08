import cv2
import numpy as np
import math
from Utils import overlayPNG

class FaceTryOn:
    def __init__(self):
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_mediapipe = True
            print("MediaPipe Initialized")
        except Exception as e:
            print(f"MediaPipe Failed, switching to OpenCV Fallback: {e}")
            # Load Haar Cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Load glasses with alpha channel
        self.glasses = cv2.imread("Resources/Glasses.png", cv2.IMREAD_UNCHANGED)
        
    def process(self, img):
        landmarks_list = []
        if self.use_mediapipe:
            return self.process_mediapipe(img)
        else:
            return self.process_opencv(img)

    def process_mediapipe(self, img):
        import mediapipe as mp
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        landmarks_list = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = img.shape
                landmarks_list = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                
                p1 = landmarks_list[130] 
                p2 = landmarks_list[359] 
                
                dX, dY = p2[0] - p1[0], p2[1] - p1[1]
                angle = math.degrees(math.atan2(dY, dX))
                dist = math.hypot(dX, dY)
                
                scale = 2.5 
                target_width = int(dist * scale)
                
                if target_width > 0:
                    try:
                        scale_factor = target_width / self.glasses.shape[1]
                        new_h = int(self.glasses.shape[0] * scale_factor)
                        resized_glasses = cv2.resize(self.glasses, (target_width, new_h))
                        
                        center = (target_width // 2, new_h // 2)
                        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        
                        rotated_glasses = cv2.warpAffine(
                            resized_glasses, M, (target_width, new_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0)
                        )
                        
                        bridge_x = (p1[0] + p2[0]) / 2
                        bridge_y = (p1[1] + p2[1]) / 2
                        
                        x_pos = int(bridge_x - target_width / 2)
                        y_pos = int(bridge_y - new_h / 2)
                        
                        img = overlayPNG(img, rotated_glasses, [x_pos, y_pos])
                    except: pass
        return img, landmarks_list

    def process_opencv(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) >= 2:
                # Sort eyes by x position
                eyes = sorted(eyes, key=lambda e: e[0])
                ex1, ey1, ew1, eh1 = eyes[0]
                ex2, ey2, ew2, eh2 = eyes[-1]
                
                # Eye center coordinates in global image
                p1 = (x + ex1 + ew1//2, y + ey1 + eh1//2)
                p2 = (x + ex2 + ew2//2, y + ey2 + eh2//2)
                
                dX, dY = p2[0] - p1[0], p2[1] - p1[1]
                angle = math.degrees(math.atan2(dY, dX))
                dist = math.hypot(dX, dY)
                
                target_width = int(dist * 3.0) # Larger factor for Haar Eyes
                if target_width > 0:
                    try:
                        scale_factor = target_width / self.glasses.shape[1]
                        new_h = int(self.glasses.shape[0] * scale_factor)
                        resized_glasses = cv2.resize(self.glasses, (target_width, new_h))
                        
                        center = (target_width // 2, new_h // 2)
                        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                        rotated_glasses = cv2.warpAffine(resized_glasses, M, (target_width, new_h),
                                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                        
                        img = overlayPNG(img, rotated_glasses, [int((p1[0]+p2[0])/2 - target_width/2), int((p1[1]+p2[1])/2 - new_h/2)])
                    except: pass
                    
        return img, []
