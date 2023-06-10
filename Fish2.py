import cv2
import numpy as np

def calculate_body_length(head_point, tail_point):
    return np.sqrt((tail_point[0] - head_point[0]) ** 2 + (tail_point[1] - head_point[1]) ** 2)

def calculate_area(contour):
    return cv2.contourArea(contour)

def calculate_depth(contour):
    _, _, _, height = cv2.boundingRect(contour)
    return height

def process_fish_image(image_path):
    # Load fish image
    image = cv2.imread(image_path)
    # Apply preprocessing steps if necessary

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to create a binary image
    _, thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Fish segmentation
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find fish contour with maximum area
    fish_contour = max(contours, key=cv2.contourArea)

    # Fish segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find fish contour with maximum area
    fish_contour = max(contours, key=cv2.contourArea)

    # Calculate fish body length
    head_point = tuple(fish_contour[0][0])
    tail_point = tuple(fish_contour[-1][0])
    body_length = calculate_body_length(head_point, tail_point)

    # Calculate other morphological parameters
    caudal_fin_length = calculate_body_length(tuple(fish_contour[10][0]), tuple(fish_contour[20][0]))
    caudal_fin_depth = calculate_depth(fish_contour)
    caudal_fin_area = calculate_area(fish_contour)
    caudal_peduncle_depth = calculate_depth(fish_contour)
    # Add calculations for other parameters (propulsive area, body area, muscle area) as per your requirements

    return {
        'body_length': body_length,
        'caudal_fin_length': caudal_fin_length,
        'caudal_fin_depth': caudal_fin_depth,
        'caudal_fin_area': caudal_fin_area,
        'caudal_peduncle_depth': caudal_peduncle_depth
        # Include other parameters in the returned dictionary
    }

# Example usage
dataset_path = '/Users/sanjkpan/Downloads/Fish_Data'
image_files = ['fish1.jpg', 'fish2.jpg', 'fish3.jpg']  # Example list of image filenames

for image_file in image_files:
    image_path = dataset_path + '/' + image_file
    result = process_fish_image(image_path)
    print(f"Fish: {image_file}")
    print(f"Body Length: {result['body_length']} units")
    print(f"Caudal Fin Length: {result['caudal_fin_length']} units")
    print(f"Caudal Fin Depth: {result['caudal_fin_depth']} units")
    print(f"Caudal Fin Area: {result['caudal_fin_area']} square units")
    print(f"Caudal Peduncle Depth: {result['caudal_peduncle_depth']} units")
    # Print other parameters as needed
    print()
