import cv2
import numpy as np
import math

class HandDetector:
    def __init__(self):
        # Initialize OpenCV's background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        
        # Parameters for hand detection
        self.min_area = 1000  # Minimum contour area to be considered a hand
        self.learning_rate = 0.001
        
        # Parameters for landmark detection
        self.max_points = 21
        self.smooth_factor = 0.5
        
        # Setup contour properties
        self.prev_contour = None
        self.landmarks = []
        
        # Initialize windows
        cv2.namedWindow('Hand Tracking')
        cv2.namedWindow('Mask')

    def detect_hand(self, frame):
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create binary mask
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5,5), 0)
        
        return mask

    def find_contours(self, mask):
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > self.min_area:
                return largest_contour
        return None

    def find_landmarks(self, contour):
        if contour is None:
            return []
        
        # Get convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Get convexity defects
        defects = cv2.convexityDefects(contour, hull)
        
        landmarks = []
        if defects is not None:
            # Get fingertips and valleys
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Add points to landmarks
                landmarks.extend([start, far])
                if i == defects.shape[0] - 1:
                    landmarks.append(end)
                    
            # Get center point
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                landmarks.append((cx, cy))
        
        return landmarks[:self.max_points]

    def draw_landmarks(self, frame, landmarks, contour):
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw convex hull
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            cv2.circle(frame, landmark, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), landmark, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def process_frame(self, frame):
        # Create a copy of the frame
        output = frame.copy()
        
        # Detect hand using skin color
        mask = self.detect_hand(frame)
        
        # Find contours
        contour = self.find_contours(mask)
        
        # Find landmarks
        if contour is not None:
            landmarks = self.find_landmarks(contour)
            self.landmarks = landmarks  # Store landmarks for smoothing
        else:
            landmarks = []
        
        # Draw visualization
        self.draw_landmarks(output, landmarks, contour)
        
        # Show the mask
        cv2.imshow('Mask', mask)
        
        return output, landmarks

def main():
    # Initialize detector
    detector = HandDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame, landmarks = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Hand Tracking', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()