import cv2
import time
import os
import numpy as np
from FaceTryOn import FaceTryOn
from WristTryOn import WristTryOn
import SkinTone

def main():
    start_app_time = time.time()
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    
    # Initialize Modules
    faceApp = FaceTryOn()
    wristApp = WristTryOn()

    mode = 'Face' 
    pTime = 0
    startup_finished = False
    startup_duration = 0
    save_feedback_timer = 0
    
    # Create Captures folder if it doesn't exist
    if not os.path.exists("Captures"):
        os.makedirs("Captures")
    
    print("Starting Virtual Try-On...")

    while True:
        success, img = cap.read()
        if not success: break
        img = cv2.flip(img, 1)
        
        # Calculate Startup Time once
        if not startup_finished:
            startup_duration = time.time() - start_app_time
            startup_finished = True

        landmarks = []
        if mode == 'Face':
            img, landmarks = faceApp.process(img)
            if landmarks:
                try:
                    recommendation, status, color = SkinTone.analyze_skin_tone(img, landmarks)
                    overlay = img.copy()
                    cv2.rectangle(overlay, (20, 100), (400, 240), (30, 30, 30), -1)
                    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
                    cv2.putText(img, "SKIN TONE ANALYSIS", (35, 130), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
                    
                    rec_color = (0, 215, 255) if recommendation == "Gold" else (220, 220, 220)
                    cv2.putText(img, f"Rec: {recommendation}", (35, 175), cv2.FONT_HERSHEY_DUPLEX, 1.2, rec_color, 2)
                    cv2.putText(img, f"Tone: {status}", (35, 215), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                    cv2.circle(img, (370, 130), 12, color, -1)
                except: pass
        elif mode == 'Wrist':
            img, landmarks = wristApp.process(img)

        # Performance Monitoring
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime
        
        # Lighting Test (Robustness)
        from Utils import get_brightness
        brightness = get_brightness(img)
        billing_status = "Good"
        status_color = (0, 255, 0)
        if brightness < 50:
            billing_status = "Too Dark"
            status_color = (0, 0, 255)
        elif brightness > 230:
            billing_status = "Too Bright"
            status_color = (0, 165, 255)

        # UI: Header & Counters
        cv2.rectangle(img, (0, 0), (1280, 70), (20, 20, 20), -1)
        cv2.putText(img, "SMART MIRROR AI", (30, 45), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
        
        # Stress Test Dashboard
        cv2.putText(img, f'FPS: {int(fps)}', (1130, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0) if fps > 20 else (0, 0, 255), 2)
        cv2.putText(img, f'Startup: {startup_duration:.2f}s', (1080, 60), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1)
        cv2.putText(img, f'Light: {billing_status}', (880, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, status_color, 2)
        
        cv2.putText(img, "[F] Face  [W] Wrist  [S] Save  [Q] Quit", (420, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 250, 150), 1)
        
        # Show "Saved!" Feedback
        if time.time() < save_feedback_timer:
            cv2.rectangle(img, (540, 300), (740, 400), (0, 200, 0), -1)
            cv2.putText(img, "SAVED!", (575, 365), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)

        cv2.imshow("Smart Mirror Evaluator", img)
        key = cv2.waitKey(1)
        if key == ord('q'): break
        elif key == ord('f'): mode = 'Face'
        elif key == ord('w'): mode = 'Wrist'
        elif key == ord('s'):
            filename = f"Captures/Snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, img)
            save_feedback_timer = time.time() + 2 # Show feedback for 2 seconds
            print(f"Captured: {filename}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
