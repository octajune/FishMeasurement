import cv2
import numpy as np
from skimage.measure import regionprops

image = cv2.imread("/Users/sanjkpan/Downloads/Fish_Data/images/raw_images/A73EGS-P_1.jpg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
edges = cv2.Canny(blurred_image, 100, 200)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(largest_contour)
fish_length = w
fish_depth = h
caudal_fin_length = regionprops(largest_contour)[0].major_axis_length
caudal_fin_depth = regionprops(largest_contour)[0].minor_axis_length
caudal_fin_area = regionprops(largest_contour)[0].area
caudal_peduncle_depth = regionprops(largest_contour)[0].convex_hull_area / caudal_fin_area
propulsive_area = caudal_fin_area - caudal_peduncle_depth * w
body_area = image.shape[0] * image.shape[1] - propulsive_area
muscle_area = body_area - caudal_fin_area

print (fish_length,fish_depth,caudal_fin_length,caudal_fin_depth,caudal_fin_area)