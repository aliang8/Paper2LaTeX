import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT
import numpy as np
from Queue import Queue
from itertools import repeat, chain, product

from graph import Graph, Node

PIXEL_UNVISITED = 0  # Value of an unvisited pixel.
PIXEL_VISITED = 120  # Value of a visited pixel.
PIXEL_DISCOVERED = 80  # Value of a discovered pixel.
PIXEL_BG = 255  # Value of a background pixel.

def xy_to_rc(point):
    return (point[1], point[0])
def rc_to_xy(point):
    return (point[1], point[0])


def get_graph(file_name, max_width=800, debug=False):
    """Given the name of an image file, generate a Graph corresponding to the
    contents of the image."""
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    # Downsample image via Gaussian pyramind.
    width, height = img.shape
    while width > max_width:
        img = cv2.pyrDown(img)
        width, height = img.shape

    # Threshold image to separate light/dark pixels.
    # TODO: Figure out appropriate threshold from the image.
    thresh_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, -5)
    """
    if thresh_img[0][0] < 255:
        thresh_img = 255 - thresh_img
    """

    img_nodes = find_circle_nodes(thresh_img, debug=debug)
    if debug:
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)

        cv2.imshow('detected circles', thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print img_nodes

    graph = find_edges(thresh_img, img_nodes, make_bbox_edge_dict(img_nodes))
    if debug:
        print graph
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)

        cv2.imshow('detected circles', thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return graph


def find_circle_nodes(img, debug=False):
    width, height = img.shape
    min_closest_dist = max(height, width) / 7

    circles = cv2.HoughCircles(img, HOUGH_GRADIENT, 1,
         min_closest_dist, param1=90, param2=25, minRadius=0, maxRadius=0)

    img_nodes = []
    for x, y, r in circles[0,:]:
        x = int(x)
        y = int(y)
        r = int(r)
        r2 = int(r + int(0.2 * r) + 5)

        if debug:
            print x, y, r, r2
        i = x
        j = y
        img_node = Node(None, bbox_tl=(max(0, i-r2), max(0, j-r2)),
            bbox_br=(min(height, i+r2), min(width, j+r2)), pos=(i, j))
        img_nodes.append(img_node)

    return img_nodes


def find_edges(image, nodes, bbox_edges):
    """Finds the edges between nodes in the given image."""

    fill_node_bboxes(image, nodes)

    # Dictionary mapping each node to its immediate neighbourhood.
    nbhds = {}

    for node in nodes:
        nbhds[node] = find_nbhd(image, nodes, bbox_edges, node)

    return make_graph(nbhds)


def fill_node_bboxes(image, nodes):
    for node in nodes:
        cv2.rectangle(image, node.bbox_tl, node.bbox_br, PIXEL_UNVISITED, -1)


def find_nbhd(image, nodes, bbox_edges, node):
    """ Finds all of the nodes that are adjacent to the given node in the image."""
    return traverse_edge(image, nodes, bbox_edges, node, node.pos)


def traverse_edge(image, nodes, bbox_edges, start_node, start_pixel):
    frontier = []
    frontier.append(xy_to_rc(start_pixel))

    found_nodes = set()

    while frontier:
        current = frontier.pop()
        image[current] = PIXEL_VISITED

        if current in bbox_edges and start_node != bbox_edges[current]: # Don't count loops as edges
            found_nodes.add(bbox_edges[current])
            continue # Don't expand current pixel if it is on the boundary of a bounding box.

        for pixel in adjacent_pixels(current, image.shape):
            if image[pixel] == PIXEL_UNVISITED:
                image[pixel] = PIXEL_DISCOVERED
                frontier.append(pixel)

    return found_nodes


def adjacent_pixels(pixel, image_shape):
    adjacent = []
    directions = product([-1, 0, 1], [-1, 0, 1])
    for dr in directions:
        if dr == (0, 0):
            continue

        adj = (pixel[0] + dr[0], pixel[1] + dr[1])
        if adj[0] >= 0 and adj[1] >= 0 and adj[0] < image_shape[0] and adj[1] < image_shape[1]:
            adjacent.append(adj)

    return adjacent


def make_bbox_iter(bbox_tl, bbox_br):
    print bbox_tl, bbox_br
    bbox_tl = (bbox_tl[1], bbox_tl[0])
    bbox_br = (bbox_br[1], bbox_br[0])

    bbox_tr = (bbox_br[0], bbox_tl[1])
    bbox_bl = (bbox_tl[0], bbox_br[1])

    tl_to_tr = zip(range(bbox_tl[0], bbox_tr[0]), repeat(bbox_tl[1])) # Top left to top right.
    tr_to_br = zip(repeat(bbox_tr[0]), range(bbox_tr[1], bbox_br[1])) # Top right to bottom right.
    br_to_bl = zip(range(bbox_br[0], bbox_bl[0], -1), repeat(bbox_br[1])) # Bottom right to bottom left.
    bl_to_tl = zip(repeat(bbox_bl[0]), range(bbox_bl[1], bbox_tl[1], -1)) # Bottom left to top left.

    return chain(tl_to_tr, tr_to_br, br_to_bl, bl_to_tl)


def make_bbox_edge_dict(nodes):
    bbox_edges = {}
    for node in nodes:
        bbox_iter = make_bbox_iter(node.bbox_tl, node.bbox_br)
        for pixel in bbox_iter:
            bbox_edges[pixel] = node

    return bbox_edges


def make_graph(nbhds):
    """Generates a Graph object from the dictionary of neighborhoods."""
    nodes = set()
    for node in nbhds.keys():
        nodes.add(Node(None, pos=node.pos, neighbors=nbhds[node]))

    return Graph(nodes)


