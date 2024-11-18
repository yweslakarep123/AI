import cv2
import mediapipe as mp
import time
import numpy as np  # Import numpy for image processing
 
class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
 
    def findHands(self, img, draw=True):
        # Increase image contrast
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)  # Adjust contrast (alpha)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
 
        bbox = None  # Initialize bounding box
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Extract landmarks to calculate bounding box
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((cx, cy))
                # Calculate bounding box
                x_vals = [pt[0] for pt in lmList]
                y_vals = [pt[1] for pt in lmList]
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                if draw:
                    # Draw landmarks
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    # Draw bounding box
                    cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
        return img, bbox
 
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        bbox = None
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)
                # Calculate bounding box
                x_vals = [lm[1] for lm in lmList]
                y_vals = [lm[2] for lm in lmList]
                x_min, x_max = min(x_vals), max(x_vals)
                y_min, y_max = min(y_vals), max(y_vals)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                if draw:
                    # Draw bounding box
                    cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
        return lmList, bbox
 
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)  # Use camera index 0 to access the default camera
    detector = handDetector()
    while True:
        success, img = cap.read()
        if not success:
            continue
        
        img, bbox = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print("Landmark 4 position:", lmList[4])
        
        # Extract and display the hand region using the bounding box
        if bbox:
            x, y, w, h = bbox
            x1 = max(x - 20, 0)
            y1 = max(y - 20, 0)
            x2 = min(x + w + 20, img.shape[1])
            y2 = min(y + h + 20, img.shape[0])
            hand_img = img[y1:y2, x1:x2]
            cv2.imshow("Hand Region", hand_img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
 
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
 
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
