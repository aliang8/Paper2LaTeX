import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT
import sys
import numpy as np
from skimage.morphology import binary_closing
from skimage.morphology import disk

if __name__ == "__main__":
    file_name = sys.argv[1]
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    width, height = img.shape
    while 900 < width:
        img = cv2.pyrDown(img)
        width, height = img.shape
    img = cv2.blur(img, (5, 5))
    cv2.imshow('blur', img)
    ret3, thresh_img_1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_img_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, -5)
    thresh_img_3 = cv2.morphologyEx(thresh_img_2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thresh_img_3_blur = cv2.blur(thresh_img_3, (8, 8))
    thresh_img_4 = cv2.erode(thresh_img_3_blur, np.ones((2, 2), np.uint8), iterations=6)
    # thresh_img_4 = cv2.morphologyEx(thresh_img_3_blur, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))
    ret3, thresh_img_4 = cv2.threshold(thresh_img_4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # mask = cv2.morphologyEx(thresh_img_2, cv2.MORPH_CLOSE, se1)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
    # thresh_img_4 = mask
cv2.imshow('opening', thresh_img_3)
cv2.imshow('img3blur', thresh_img_3_blur)
cv2.imshow('final', thresh_img_4)
cv2.waitKey(0)
cv2.destroyAllWindows()