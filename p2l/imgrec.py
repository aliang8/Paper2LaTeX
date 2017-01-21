import cv2
#import cv2.cv as cv
import numpy as np
from Queue import Queue
from itertools import repeat, chain, product

from graph import Graph, Node

PIXEL_UNVISITED = 255  # Value of an unvisited pixel.
PIXEL_VISITED = 120  # Value of a visited pixel.
PIXEL_DISCOVERED = 80  # Value of a discovered pixel.
PIXEL_BG = 0  # Value of a background pixel.

def get_graph(file_name, debug=False):
    """Given the name of an image file, generate a Graph corresponding to the
    contents of the image."""
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    # Downsample image via Gaussian pyramind.
    width, height = img.shape
    while 1200 < width:
        img = cv2.pyrDown(img)
        width, height = img.shape

    # Threshold image to separate light/dark pixels.
    # TODO: Figure out appropriate threshold from the image.
    _, thresh_img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
    """
    if thresh_img[0][0] < 255:
        thresh_img = 255 - thresh_img
    """

    img_nodes = find_circle_nodes(thresh_img, debug=debug)

    print img_nodes

    return img_nodes


def find_circle_nodes(img, debug=False):
    width, height = img.shape
    min_closest_dist = max(height, width) / 7
    '''
    circles = cv2.HoughCircles(img, cv.CV_HOUGH_GRADIENT, 1,
         min_closest_dist, param1=90, param2=25, minRadius=0, maxRadius=0)
    '''
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
         min_closest_dist, param1=90, param2=25, minRadius=0, maxRadius=0)

    img_nodes = []
    for x, y, r in circles[0,:]:
        x = int(x)
        y = int(y)
        r = int(r)
        r2 = int(r + int(0.2 * r) + 5)

        if debug:
            print x, y, r, r2
            # draw the outer circle
            cv2.circle(img,(x,y),r,(100,255,100),2)
            # draw the center of the circle
            cv2.circle(img,(x,y),2,(100,100,255),3)
            # draw bounding box
            cv2.rectangle(img, (max(0, x-r2), max(0, y-r2)), (min(width, x+r2), min(height, y+r2)), (125, 125, 25), 2)
        img_node = Node((max(0, x-r2), max(0, y-r2)), (min(width, x+r2), min(height, y+r2)), x, y)
        img_nodes.append(img_node)

    if debug:
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)

        cv2.imshow('detected circles', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_nodes



