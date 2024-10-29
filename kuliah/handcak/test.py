import cv2
import time
import handTrackingModule as htm
import serial

# Configure your serial port to connect with Arduino (replace 'COM4' with your port)
arduino = serial.Serial('COM4', 115200, timeout=1)

def getNumber(ar):
    s = ""
    for i in ar:
        s += str(ar[i])
       
    if s == "00000":
        return 0
    elif s == "01000":
        return 1
    elif s == "01100":
        return 2 
    elif s == "01110":
        return 3
    elif s == "01111":
        return 4
    elif s == "11111":
        return 5
    elif s == "01001":
        return 6
    elif s == "01011":
        return 7
    return -1  # Return -1 if no valid number is recognized
 
wcam, hcam = 640, 480
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)
    tipId = [4, 8, 12, 16, 20]
    
    detectedNumber = -1

    if len(lmList) != 0:
        fingers = []
        
        # Thumb
        if lmList[tipId[0]][1] > lmList[tipId[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, len(tipId)):
            if lmList[tipId[id]][2] < lmList[tipId[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        detectedNumber = getNumber(fingers)
        
        # Draw the detected number on the screen
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)   
        cv2.putText(img, str(detectedNumber), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 20)  
        
        # Send the detected number to the Arduino
        if detectedNumber != -1:
            print(f"Detected number: {detectedNumber}")  # Print detected number
            arduino.write(bytes(str(detectedNumber), 'utf-8'))  # Send number as a single byte
            time.sleep(0.2)  # Add a small delay to allow Arduino to process data
            data = arduino.readline()
            print(data)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS on the screen
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

    # Display the image
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
