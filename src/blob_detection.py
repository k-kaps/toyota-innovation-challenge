import cv2
import numpy as np

def verticalHoleDetection(img_original, img):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 0.3
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("", im_with_keypoints)
    cv2.waitKey(0) 

def improperHoleDetection_small(img_original, img):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 3000
    params.maxArea = 4000
    params.filterByCircularity = True
    params.minCircularity = 0.1
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0) 

def properHoleDetection_horizontal(img, parameters):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 1500
    params.minArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.maxCircularity= parameters[4]
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.maxConvexity = 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.4
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)    
    
def improperHoleDetection_large(img, parameters):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = parameters[0]
    params.maxThreshold = parameters[1]
    params.filterByArea = True
    params.minArea = parameters[2]
    params.maxArea = parameters[3]
    params.filterByCircularity = True
    params.minCircularity = parameters[4]
    params.maxCircularity = parameters[5]
    params.filterByConvexity = True
    params.minConvexity = parameters[6]
    params.maxConvexity = parameters[7]
    params.filterByInertia = True
    params.minInertiaRatio = parameters[8]
    params.maxInertiaRatio = parameters[9]
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), parameters[10], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

def blobDetection(img_original, img, parameters):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = parameters[0]
    params.maxThreshold = parameters[1]
    params.filterByArea = True
    params.minArea = parameters[2]
    params.maxArea = parameters[3]
    params.filterByCircularity = True
    params.minCircularity = parameters[4]
    params.maxCircularity = parameters[5]
    params.filterByConvexity = True
    params.minConvexity = parameters[6]
    params.maxConvexity = parameters[7]
    params.filterByInertia = True
    params.minInertiaRatio = parameters[8]
    params.maxInertiaRatio = parameters[9]
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else : 
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    
    im_with_keypoints = cv2.drawKeypoints(img_original, keypoints, np.array([]), parameters[10], cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

    return im_with_keypoints

def circleDetection(img_original, img):
    rows = img.shape[0]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, rows / 8, param1 = 120, param2 = 20, minRadius=0, maxRadius=150)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 0, 0), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)
    
    cv2.imshow("RESULTS", img)
    cv2.waitKey()
    

def preprocessing(filepath):
    img = cv2.imread(filepath)
    img_original = img.copy()
    img = cv2.GaussianBlur(img, (3, 3), 5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges, img = cv2.threshold(img, 80, 255, cv2.THRESH_TRUNC)
    img_thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    cv2.imshow("RESULTS", img)
    cv2.imshow("ORIGINAL", img_thresh2)
    cv2.waitKey()
    # PARAMS: [minThreshold, maxThreshold, minArea, maxArea, minCircularity, maxCircularity, minConvexity, maxConvexity,
    #          minInertiaRatio, maxInertiaRatio, color_of_circle]
    params1 = [0, 100, 5000, 1000000, 0.01, 1, 0.5, 1, 0.1, 1, (0, 0, 255)] #FOR LARGE & IMPROPER H.D.
    params2 = [0, 200, 1500, 3150, 0.8, 1, 0.01, 1, 0.45, 1, (0, 255, 0)] #FOR NORMAL & PROPER H.D.
    params3 = [0, 200, 2650, 5000, 0.1, 0.9, 0.01, 1, 0.1, 1, (0, 255, 255)] #FOR NORMAL & IMPROPER H.D.
    params4 = [0, 200, 500, 1700, 0.7, 1, 0.8, 1, 0.01, 0.4, (255, 0, 0)] #FOR VERTICAL & PROPER H.D.
    params5 = [0, 200, 1700, 2000, 0.01, 0.89, 0.8, 1, 0.01, 1, (255, 0, 255)] #FOR VERTICAL & IMPROPER H.D.

    img_original = blobDetection(img_original, img, params1)
    img_original = blobDetection(img_original, img, params2)
    img_original = blobDetection(img_original, img, params3)
    img_original = blobDetection(img_original, img, params4)
    img_original = blobDetection(img_original, img, params5)
    #improperHoleDetection_small(img_original, img)
    #properHoleDetection(img_original, img)
    #verticalHoleDetection(img_original, img)

if __name__ == '__main__':
    i = input("Enter the file name: ")
    filepath = "data/Metal_" + str(i) + ".jpg"
    preprocessing(filepath)
        
