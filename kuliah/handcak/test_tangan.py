import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define ROI (Region of Interest)
    roi = frame[100:400, 300:600]
    cv2.rectangle(frame, (300, 100), (600, 400), (0, 255, 0), 2)

    # Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply Gaussian blur
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply morphological transformations
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.erode(mask, kernel, iterations=2)

    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Proceed if contours are found
    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Draw the contour on the ROI
        cv2.drawContours(roi, [max_contour], -1, (255, 0, 0), 2)

        # Find the convex hull
        hull = cv2.convexHull(max_contour)
        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)

        # Find convexity defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(max_contour, hull_indices)

            # Analyze defects to detect fingertips
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(max_contour[s][0])
                    end = tuple(max_contour[e][0])
                    far = tuple(max_contour[f][0])

                    # Convert points to NumPy arrays
                    start_array = np.array(start)
                    end_array = np.array(end)
                    far_array = np.array(far)

                    # Calculate the distances between points
                    a = np.linalg.norm(end_array - start_array)
                    b = np.linalg.norm(far_array - start_array)
                    c = np.linalg.norm(end_array - far_array)

                    # Calculate the angle using the cosine rule
                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * (180 / np.pi)

                    # Ignore angles greater than 90 degrees and highlight fingertips
                    if angle <= 90:
                        cv2.circle(roi, far, 5, [0, 0, 255], -1)
                        cv2.line(roi, start, end, [0, 255, 0], 2)

    # Display the frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
