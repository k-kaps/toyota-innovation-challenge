import cv2
import numpy as np
import matplotlib.pyplot as plt

#GLOBAL PATH
PATH = "TMMC-IMAGES/"

#READING IMAGE
img_1 = cv2.imread(PATH + "Metal_2.jpg")
#img_2 = cv2.imread(PATH + "Metal_6.jpg")
img_1 = cv2.resize(img_1, (1200, 600))
img_1_copy = img_1.copy()
img_1 = cv2.GaussianBlur(img_1, (7, 7), 5)
#img_2 = cv2.GaussianBlur(img_2, (7, 7), 5)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_RGB2GRAY)
#img_2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2GRAY)
img_thresh1 = cv2.adaptiveThreshold(img_1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_MASK, 9, 2)
#img_thresh2 = cv2.adaptiveThreshold(img_2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
#edges_1 = cv2.Canny(img_thresh1, 60, 255)
#kernel = np.ones((1, 1), np.uint8)
#edges_1 = cv2.dilate(edges_1, kernel, iterations=1)
#cv2.imshow("RESULTS", edges_1)
#edges_2 = cv2.Canny(img_thresh2, 10, 255)
params = cv2.SimpleBlobDetector_Params()
  
# Set Area filtering parameters
params.filterByArea = True
params.minArea = 300
  
# Set Circularity filtering parameters
params.filterByCircularity = True 
params.maxCircularity = 0.95
params.minCircularity = 0.55
  
# Set Convexity filtering parameters
params.filterByConvexity = True
params.minConvexity = 0.2
      
# Set inertia filtering parameters
params.filterByInertia = True
params.maxInertiaRatio = 0.99
params.minInertiaRatio = 0.01
  
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
      
# Detect blobs
keypoints = detector.detect(img_thresh1)
  
# Draw blobs on our image as red circles
blank = np.zeros((1, 1)) 
blobs = cv2.drawKeypoints(img_1_copy, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  
number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
  
# Show blobs
cv2.imshow("RESULTS 1", blobs)
cv2.imshow("RESULTS 2", img_thresh1)
cv2.waitKey()
cv2.destroyAllWindows()