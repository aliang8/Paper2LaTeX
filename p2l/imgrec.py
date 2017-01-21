import cv2
import cv2.cv as cv
import numpy as np

from graph import Graph, Node

def get_semantics(file_name):
    """Given the name of an image file, generate a Graph corresponding to the
    contents of the image."""
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    # Downsample image via Gaussian pyramind.
    width, height = img.shape
    while 1200 < width:
        img = cv2.pyrDown(img)
        width, height = img.shape

    # TODO: Figure out what this means.


    # Threshold image to separate light/dark pixels.
    # TODO: Figure out appropriate threshold from the image.
    _, thresh_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    """
    if thresh_img[0][0] < 255:
        thresh_img = 255 - thresh_img
    """

    """
    cv2.imshow("image", thresh_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    img_nodes = find_nodes(thresh_img)


def find_nodes(img):
    width, height = img.shape
    min_closest_dist = max(height, width) / 7
    circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 1,
         min_closest_dist, param1=100, param2=30, minRadius=0, maxRadius=0)

    print circles

