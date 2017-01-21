import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT
import sys

if __name__ == "__main__":
    file_name = sys.argv[1]
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    width, height = img.shape
    while 900 < width:
        img = cv2.pyrDown(img)
        width, height = img.shape
    ret3, thresh_img_1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_img_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, -5)

cv2.imshow('image normal', thresh_img_1)
cv2.imshow('image_adaptive', thresh_img_2)
cv2.waitKey(0)
cv2.destroyAllWindows()