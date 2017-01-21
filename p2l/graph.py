class Graph():
    def __init__(self, nodes, directed=False):
        self.nodes = nodes
        self.directed = directed

    def __repr__(self):
        result = ""
        for node in self.nodes:
            result += "%s -- %s\n" % (node.__repr__(), node.neighbors.__repr__())
        return result

class Node():
    """ A node in a graph. Each node stores information about its (x, y)
    coordinates, as well as a set of its neighbours. """
    def __init__(self, shape, bbox_tl=(-1, -1), bbox_br=(-1, -1), x_pos=-1, y_pos=-1,
            neighbors={}):
        self.bbox_tl = bbox_tl
        self.bbox_br = bbox_br
        self.neighbors = neighbors
        self.x_pos = x_pos
        self.y_pos = y_pos

    def __hash__(self):
        return hash((self.x_pos, self.y_pos))

    def __repr__(self):
        return "Node(%d, %d)" % (self.x_pos, self.y_pos)

    def __eq__(self, other):
        return self.x_pos == other.x_pos and self.y_pos == other.y_pos

