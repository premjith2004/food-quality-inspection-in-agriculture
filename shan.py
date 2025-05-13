import cv2
import numpy as np

def inspect_food_quality():
    # Load the image
    image = cv2.imread(image_path)
    original = image.copy()
    image = cv2.resize(image, (640, 480))

    # Convert to HSV color space for better color filtering
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for healthy color (e.g., red for tomatoes)
    lower_healthy = np.array([0, 100, 100])
    upper_healthy = np.array([10, 255, 255])
    healthy_mask1 = cv2.inRange(hsv, lower_healthy, upper_healthy)

    lower_healthy2 = np.array([160, 100, 100])
    upper_healthy2 = np.array([180, 255, 255])
    healthy_mask2 = cv2.inRange(hsv, lower_healthy2, upper_healthy2)

    healthy_mask = healthy_mask1 | healthy_mask2

    # Invert to get potential defect areas
    defect_mask = cv2.bitwise_not(healthy_mask)

    # Apply morphology to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the defects
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw defect areas on the original image
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Only consider significant defects
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

    # Display results
    cv2.imshow("Original", original)
    cv2.imshow("Healthy Mask", healthy_mask)
    cv2.imshow("Defect Mask", defect_mask)
    cv2.imshow("Defect Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
inspect_food_quality("tomato_sample.jpg")
