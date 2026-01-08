import cv2
import time
import numpy as np
from FaceTryOn import FaceTryOn
from WristTryOn import WristTryOn
import SkinTone

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    # Initialize Modules with built-in fallbacks
    faceApp = FaceTryOn()
    wristApp = WristTryOn()

    mode = 'Face' 
    pTime = 0
    
    print("Starting Virtual Try-On...")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        
        landmarks = []
        if mode == 'Face':
            img, landmarks = faceApp.process(img)
            if landmarks:
                try:
                    recommendation, color = SkinTone.analyze_skin_tone(img, landmarks)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (20, 100), (350, 220), (30, 30, 30), -1)
                    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                    cv2.putText(img, "SKIN TONE ANALYSIS", (35, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
                    rec_color = (0, 215, 255) if recommendation == "Gold" else (192, 192, 192)
                    cv2.putText(img, f"{recommendation}", (35, 180), cv2.FONT_HERSHEY_DUPLEX, 1.5, rec_color, 2)
                    cv2.circle(img, (320, 130), 15, color, -1)
                except: pass
        elif mode == 'Wrist':
            img, landmarks = wristApp.process(img)

        # UI: Header
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        cv2.rectangle(img, (0, 0), (1280, 60), (0, 0, 0), -1)
        cv2.putText(img, "VIRTUAL TRY-ON AI", (50, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        cv2.putText(img, f'FPS: {int(fps)}', (1150, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.putText(img, "[F] Face  [W] Wrist  [Q] Quit", (400, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 1)

        cv2.imshow("Virtual Try-On Project", img)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == ord('f'): mode = 'Face'
        elif key == ord('w'): mode = 'Wrist'
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
