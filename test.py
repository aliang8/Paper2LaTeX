import cv2
from p2l import *
from p2l import edge


imgPath = 'examples/2nodes_dark.jpg'

nodes = get_graph(imgPath, True)


bbox_edge = edge.make_bbox_edge_dict(nodes)

edge.find_edges(cv2.imread(imgPath), nodes, bbox_edge)
