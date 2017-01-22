import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT
import numpy as np
from itertools import repeat, chain, product
from Queue import Queue
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from graph import Graph, Node
import math

PIXEL_UNVISITED = 0  # Value of an unvisited pixel.
PIXEL_VISITED = 120  # Value of a visited pixel.
PIXEL_DISCOVERED = 80  # Value of a discovered pixel.
PIXEL_BG = 255  # Value of a background pixel.

def xy_to_rc(point):
    return (point[1], point[0])
def rc_to_xy(point):
    return (point[1], point[0])


def get_graph(file_name, max_width=900, debug=False):
    """Given the name of an image file, generate a Graph corresponding to the
    contents of the image."""
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # Downsample image via Gaussian pyramind.
    width, height = img.shape
    while width > max_width:
        img = cv2.pyrDown(img)
        width, height = img.shape

    img = cv2.blur(img, (5, 5))
    cv2.imshow('blur', img)
    _, thresh_img_1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_img_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, -5)
    thresh_img_3 = cv2.morphologyEx(thresh_img_2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    thresh_img_3_blur = cv2.blur(thresh_img_3, (8, 8))
    thresh_img_4 = cv2.erode(thresh_img_3_blur, np.ones((2, 2), np.uint8), iterations=3)
    _, thresh_img_5 = cv2.threshold(thresh_img_4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_img_6 = img_as_ubyte(skeletonize(thresh_img_5 != 255))
    _, thresh_img_7 = cv2.threshold(thresh_img_6, 0, 255, cv2.THRESH_BINARY_INV)

    thresh_img = thresh_img_7
    thresh_img_edges = thresh_img_7

    img_nodes = find_circle_nodes(thresh_img, debug=debug)
    if debug:
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)

        cv2.imshow('detected circles', thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print img_nodes

    graph = find_edges(thresh_img_7, img_nodes, make_bbox_edge_dict(thresh_img_7, img_nodes))
    if debug:
        print graph
        cv2.namedWindow('detected circles', cv2.WINDOW_NORMAL)

        cv2.imshow('detected circles', thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return graph


def find_circle_nodes(img, debug=False, minRadius=5, maxRadius=50):
    height, width = img.shape
    min_closest_dist = float(max(height, width)) / 5.5

    circles = cv2.HoughCircles(img, HOUGH_GRADIENT, 1.2,
         min_closest_dist, param1=80, param2=25, minRadius=minRadius, maxRadius=maxRadius)

    img_nodes = []
    if circles is not None:
        for x, y, r in circles[0,:]:
            print x,y,r
            x = int(x)
            y = int(y)
            r = int(r)
            r2 = int(r * 1.2 + 5)

            if debug:
                print x, y, r, r2
            i = x
            j = y

            img_circle = img[max(0, j-r2):min(height, j+r2), max(0, i-r2):min(width, i+r2)]
            
            if cv2.HoughCircles(img_circle, HOUGH_GRADIENT, 1,
                    min_closest_dist, param1=50, param2=10, minRadius=minRadius, maxRadius=maxRadius) is not None:
                img_node = Node(None, bbox_tl=(max(0, i - r2), max(0, j - r2)),
                    bbox_br=(min(width, i + r2), min(height, j + r2)), pos=(i, j))
                img_nodes.append(img_node)

    return img_nodes


def find_edges(image, nodes, bbox_edges):
    """Finds the edges between nodes in the given image."""

    fill_node_bboxes(image, nodes)

    # Dictionary mapping each node to its immediate neighbourhood.
    nbhds = {}

    for node in nodes:
        nbhds[node] = find_nbhd(image, nodes, bbox_edges, node)

    return make_graph(nodes, nbhds)


def fill_node_bboxes(image, nodes, debug=False, color=PIXEL_BG):
    for node in nodes:
        if debug:
            color = 125
        cv2.rectangle(image, node.bbox_tl, node.bbox_br, color, -1)
    if debug:
        cv2.imshow("fill_node_bboxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def find_nbhd(image, nodes, bbox_edges, node):
    """ Finds all of the nodes that are adjacent to the given node in the image."""
    height, width = image.shape
    bbox_iter_outer = make_bbox_iter(image, node.bbox_tl, node.bbox_br, resize=1)
        #(max(0, node.bbox_tl[0] - 2), max(0, node.bbox_tl[1] - 2)),
        #(min(height - 1, node.bbox_br[0] + 2), min(width - 1, node.bbox_br[1] + 2)))

    start_pixels = []
    on_edge = False
    for pixel in bbox_iter_outer:
        if on_edge:
            if image[pixel] == PIXEL_BG:
                on_edge = False
        else:
            if image[pixel] == PIXEL_UNVISITED:
                start_pixels.append(pixel)
                on_edge = True

    print "start pixels", start_pixels

    bbox_iter_inner = make_bbox_iter(image, node.bbox_tl, node.bbox_br)
    for pixel in bbox_iter_inner:
        image[pixel] = PIXEL_BG

    nbhd = set()
    for start_pixel in start_pixels:
        nbhd.update(traverse_edge(image, nodes, bbox_edges, node, start_pixel))

    return nbhd

def traverse_edge(image, nodes, bbox_edges, start_node, start_pixel, lookback=4):
    height, width = image.shape

    # Reset all non-background pixels. Quick-and-dirty fix for issues related
    # to crossing edges. TODO: Find a faster way to achieve same effect.
    for i in range(height):
        for j in range(width):
            if image[i, j] != PIXEL_BG:
                image[i, j] = PIXEL_UNVISITED

    frontier = []
    frontier.append(start_pixel)

    found_nodes = set()

    checkpoints = []

    iter = 0
    while frontier:
        current = frontier.pop()
        image[current] = PIXEL_VISITED

        checkpoints.insert(0, current)
        c_step = lookback

        if current in bbox_edges and start_node != bbox_edges[current]: # Don't count loops as edges
            found_nodes.add(bbox_edges[current])
            return found_nodes
            # Don't expand current pixel if it is on the boundary of a bounding box.

        next_pixels = adjacent_pixels(current, image.shape)
        if len(checkpoints) >= lookback:
            checkpoints.pop()
            c_end = checkpoints[0]
            c_start = checkpoints[len(checkpoints) - 1]
            c_next_y, c_next_x = current[0], current[1]
            if c_end[0] > c_start[0]:
                c_next_y += c_step
            elif c_end[0] < c_start[0]:
                c_next_y -= c_step
            if c_end[1] > c_start[1]:
                c_next_x += c_step
            elif c_end[1] < c_start[1]:
                c_next_x -= c_step
            c_next_hat = (c_next_y, c_next_x)
            # Enforce that the point closest to checkpt_3 be explored first.
            neg_dists = [-distance(px, c_next_hat) for px in next_pixels]
            next_pixels = [px for _, px in sorted(zip(neg_dists, next_pixels))]

        for pixel in next_pixels:
            if image[pixel] == PIXEL_UNVISITED:
                image[pixel] = PIXEL_DISCOVERED
                frontier.append(pixel)

        iter += 1

    return found_nodes


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (b[1] - a[1]) ** 2)

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


def make_bbox_iter(image, bbox_tl, bbox_br, resize=0):
    height, width = image.shape
    print bbox_tl, bbox_br
    bbox_tl = (max(0, bbox_tl[1] - resize), max(0, bbox_tl[0] - resize))
    bbox_br = (min(height - 1, bbox_br[1] + resize), min(width - 1, bbox_br[0] + resize))

    bbox_tr = (bbox_br[0], bbox_tl[1])
    bbox_bl = (bbox_tl[0], bbox_br[1])

    tl_to_tr = zip(range(bbox_tl[0], bbox_tr[0]), repeat(bbox_tl[1])) # Top left to top right.
    tr_to_br = zip(repeat(bbox_tr[0]), range(bbox_tr[1], bbox_br[1])) # Top right to bottom right.
    br_to_bl = zip(range(bbox_br[0], bbox_bl[0], -1), repeat(bbox_br[1])) # Bottom right to bottom left.
    bl_to_tl = zip(repeat(bbox_bl[0]), range(bbox_bl[1], bbox_tl[1], -1)) # Bottom left to top left.

    return chain(tl_to_tr, tr_to_br, br_to_bl, bl_to_tl)


def make_bbox_edge_dict(image, nodes):
    bbox_edges = {}
    for node in nodes:
        bbox_iter = make_bbox_iter(image, node.bbox_tl, node.bbox_br, resize=2)
        for pixel in bbox_iter:
            bbox_edges[pixel] = node

    return bbox_edges


def make_graph(nodes, nbhds):
    """Generates a Graph object from the dictionary of neighborhoods."""
    new_nodes = set()
    old_to_new = {}
    new_to_old = {}
    for node in nbhds.keys():
        new_node = Node(None, pos=node.pos)
        new_nodes.add(new_node)
        old_to_new[node] = new_node
        new_to_old[new_node] = node

    for new_node in new_nodes:
        nbhd = set(old_to_new[nbr] for nbr in nbhds[new_to_old[new_node]])
        new_node.neighbors = nbhd

    return Graph(new_nodes)


