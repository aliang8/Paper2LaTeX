import cv2
from p2l import get_graph
from p2l import find_edge

imgPath = 'examples/2nodes_dark.jpg'

nodes = get_graph(imgPath, True)

find_edges(cv2.imread(imgPath), nodes, )
