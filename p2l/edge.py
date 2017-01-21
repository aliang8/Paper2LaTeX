import numpy as np
from Queue import Queue
from itertools import repeat, chain, product

import cv2

#from graph.graph import Graph, Node

PIXEL_UNVISITED = 255 # Value of an unvisited pixel.
PIXEL_VISITED = 120 # Value of a visited pixel.
PIXEL_DISCOVERED = 80 # Value of a discovered pixel.
PIXEL_BG = 0 # Value of a background pixel.


# How does find_unvisited_out_srcs(image, node) work?


# In order to get pixels of a bounding box
def make_bbox_iter(bbox_tl, bbox_br):

    print bbox_tl
    print bbox_br

    bbox_tl = (bbox_tl[1], bbox_tl[0])
    bbox_br = (bbox_br[1], bbox_br[0])

    bbox_tr = (bbox_br[0], bbox_tl[1])
    bbox_bl = (bbox_tl[0], bbox_br[1])

    tl_to_tr = zip(range(bbox_tl[0], bbox_tr[0]), repeat(bbox_tl[1])) # Top left to top right.
    tr_to_br = zip(repeat(bbox_tr[0]), range(bbox_tr[1], bbox_br[1])) # Top right to bottom right.
    br_to_bl = zip(range(bbox_br[0], bbox_bl[0], -1), repeat(bbox_br[1])) # Bottom right to bottom left.
    bl_to_tl = zip(repeat(bbox_bl[0]), range(bbox_bl[1], bbox_tl[1], -1)) # Bottom left to top left.

    return chain(tl_to_tr, tr_to_br, br_to_bl, bl_to_tl)



# In order to create list of all pixels of every bounding box
def make_bbox_edge_dict(nodes):
    bbox_edges = {}
    for node in nodes:
        bbox_iter = make_bbox_iter(node.bbox_tl, node.bbox_br)
        for pixel in bbox_iter:
            bbox_edges[pixel] = node

    return bbox_edges



# Trying to find the neighborhood pixels
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



def find_unvisited_out_srcs(image, node):
    """ Returns a set of pixel representatives of unvisited edges incident to the given node. """
    bbox_iter = make_bbox_iter(node.bbox_tl, node.bbox_br)

    print "BBOX COORDS"
    print node.bbox_tl, node.bbox_br

    reps = []         # Represented 
    cur_rep = None    # What is cur_rep?


    # For every pixel in the bounding box
    for pixel in bbox_iter:

	# If the pixel we are at is unvisited
		# 
        if image[pixel] == PIXEL_UNVISITED:
            if not cur_rep:
                cur_rep = pixel
                reps.append(pixel)
        else:
            cur_rep = None

    return reps



def find_nbhd(image, nodes, bbox_edges, node):
    """ Finds all of the nodes that are adjacent to the given node in the image.
    image -- a 2d Numpy ndarray
    nodes -- a list of the nodes detected in the image
    bbox_edges -- a dictionary indicating whether a pixel at a given coordinate is on the edge of a node's bounding box
    node -- the node whose edges we wish to find
    """
    #start = node.rep_pixel # Some representative pixel that is a black pixel on the node.
    nbhd = set()


    #### What impact does out_srcs have? We don't even use outreps as it is already commented out#######

    # Find clusters of unvisited pixels along the border of the bounding box;
    # these will be the starting points for traversing the pixels in each edge.
    out_srcs = find_unvisited_out_srcs(image, node)
    # out_srcs = 

    """
    print "out_srcs: ", out_srcs, node
    for pixel in out_srcs:
        found_nodes = traverse_edge(image, nodes, bbox_edges, node, pixel)
        nbhd = nbhd.union(found_nodes)
    """

    # Once we are done, we have extended to all connected nodes of our SINGLE current node
    # Returns a list of nodes connected to our SINGLE current node
    nbhd = traverse_edge(image, nodes, bbox_edges, node, (node.y_pos, node.x_pos))

    print "nbhd: ", nbhd

    return nbhd



# Traversing edges via breath first search
def traverse_edge(image, nodes, bbox_edges, node, start_pixel):


    # This queue is suppose to store x and y position of a pixel
    frontier = Queue()

    # start_pixel is a pixel that is at the intersection of bounding box and the node's edge
    frontier.put(start_pixel)

    found_nodes = set()

    while not frontier.empty():
	# get current pixel we are on and mark it as visited
        current = frontier.get()
        #print current, image[current]
        image[current] = PIXEL_VISITED

	# Detect if current pixel is actually the bounding box AND that the current pixel is the NOT the starting pixel
	# If so, we declare that we found a node
        if current in bbox_edges and node != bbox_edges[current]: # Don't count loops as edges
            found_nodes.add(bbox_edges[current])
            continue # Don't expand current pixel if it is on the boundary of a bounding box.


	# Get pixels that is adjacent to the current pixel     (what does image.shape do?)
        for pixel in adjacent_pixels(current, image.shape):

	    # If the adjacent pixel was not visited:
		# Declare the pixel as discovered
	    	# Add a pixel to the queue so that we eventually traverse to it	
            if image[pixel] == PIXEL_UNVISITED:
                image[pixel] = PIXEL_DISCOVERED
                frontier.put(pixel)

    return found_nodes





def fill_node_bboxes(image, nodes):

    # traversing through all nodes
    for node in nodes:

        # declaring two coordinates of bounding box
        bbox_tl = node.bbox_tl
        bbox_br = node.bbox_br


        # I assume this to iterate through all pixels of bounding box
        for i in range(bbox_tl[0], bbox_br[0] + 1):
            for j in range(bbox_tl[1], bbox_br[1] + 1):

		# Declare all pixels in bounding box as UNVISITED
                image[j, i] = PIXEL_UNVISITED



# The main function that starts it all
def find_edges(image, nodes, bbox_edges):
    """ Finds the edges between nodes in the given image.
    image -- a 2d Numpy ndarray
    nodes -- a list of the nodes detected in the image
    bbox_edges -- a dictionary indicating whether a pixel at a given coordinate is on the edge of a node's bounding box
    """
    # Declare all pixels in bounding box as UNVISITED
    fill_node_bboxes(image, nodes)

    # Dictionary mapping each node to its immediate neighbourhood.
    nbhds = {}
  

    # This is to preform BFS on each node
    for node in nodes:

	# Preform BFS on a single node
	# Returns a list of nodes connected to our SINGLE current node
        nbhds[node] = find_nbhd(image, nodes, bbox_edges, node)


    # Lets not worry about this right now
    #return make_graph(nbhds)




######### LET'S NOT WORRY ABOUT THIS RIGHT NOW ############
'''
def make_graph(nbhds):
    #""" Generates a Graph object from the dictionary of neighborhoods. """
    nodes = set()
    for node in nbhds.keys():
        nodes.add(Node(x_pos=node.x_pos, y_pos=node.y_pos, neighbors=nbhds[node]))

    return Graph(nodes)
'''
