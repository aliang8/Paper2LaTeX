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
    def __init__(self, shape, bbox_tl=None, bbox_br=None, pos=None, neighbors={}):
        self.bbox_tl = bbox_tl
        self.bbox_br = bbox_br
        self.neighbors = neighbors
        self.pos = pos

    def __hash__(self):
        return hash(self.pos)

    def __repr__(self):
        if self.pos is not None:
            return "Node(%d, %d)" % (self.pos[0], self.pos[1])
        return "Node(None)"

    def __eq__(self, other):
        return self.pos == other.pos

