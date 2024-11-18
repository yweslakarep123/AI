import cv2
import time
import handTrackingModule as htm
# import serial

# Configure your serial port to connect with Arduino (replace 'COM4' with your port)
# arduino = serial.Serial('COM4', 115200, timeout=1)

def getNumber(ar):
    s = ""
    for i in ar:
        s += str(i)
       
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
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cap.read()
    if not success:
        continue

    # Call findHands and handle both img and bbox
    img, bbox = detector.findHands(img, draw=True)
    # Ensure img is an image (NumPy array)
    if img is None:
        continue

    # Call findPosition and get lmList and bbox
    lmList, bbox = detector.findPosition(img, draw=False)
    tipIds = [4, 8, 12, 16, 20]
    
    detectedNumber = -1

    if len(lmList) != 0:
        # Create a dictionary to map landmark IDs to their coordinates
        lmDict = {lm[0]: (lm[1], lm[2]) for lm in lmList}

        fingers = []
        
        # Thumb
        if 4 in lmDict and 3 in lmDict:
            # Comparing x-coordinates for thumb
            if lmDict[4][0] > lmDict[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in [8, 12, 16, 20]:
            if id in lmDict and (id - 2) in lmDict:
                # Comparing y-coordinates for fingers
                if lmDict[id][1] < lmDict[id - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                fingers.append(0)

        detectedNumber = getNumber(fingers)
        
        # Draw the detected number on the screen
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)   
        cv2.putText(img, str(detectedNumber), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 20)  

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)  # Added small value to prevent division by zero
    pTime = cTime

    # Display FPS on the screen
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

    # Display the image
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# arduino.close()
