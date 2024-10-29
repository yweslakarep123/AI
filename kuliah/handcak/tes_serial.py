import cv2
import time
import handTrackingModule as htm
import serial

# Make sure to use the correct COM port
arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # Wait for connection

wcam, hcam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
detector = htm.handDetector(detectionCon=0.75)

last_sent_time = 0
delay_between_sends = 0.1  # 100ms delay between sends

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    
    current_time = time.time()
    
    if len(lmList) != 0 and (current_time - last_sent_time) >= delay_between_sends:
        fingers = []
        
        # Thumb
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other fingers
        for id in [8, 12, 16, 20]:
            if lmList[id][2] < lmList[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # Count raised fingers
        totalFingers = fingers.count(1)
        print(f"Detected fingers: {totalFingers}")
        
        # Send to Arduino with newline
        arduino.write(f"{totalFingers}\n".encode())
        time.sleep(0.2)
        last_sent_time = current_time
        
        # Draw rectangle and number
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 20)
    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()