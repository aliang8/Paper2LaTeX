from graph import *

node1 = Node(None, pos=(10,5))
node2 = Node(None, pos=(12,1))
node3 = Node(None, pos=(3,4))
node4 = Node(None, pos=(4,9))
node1.neighbors = set([node2, node3, node4])
node2.neighbors = set([node1,node3])
node3.neighbors = set([node1])
graph = Graph({node1, node2, node3, node4})

def transpile(g):
    color="red!20"
    shape="circle"
    size="1.5cm"

    def getNodeString(node):
      return "({0},{1}) node[{2}, fill={3}, minimum size={4}] {5};\n".format(node.pos[0], -1 * node.pos[1], shape, color, size, "{}")
    def getEdgeString(node1, node2):
      return "({0},{1}) -- ({2},{3});\n".format(node1.pos[0], -1 * node1.pos[1], node2.pos[0], -1 * node2.pos[1])

    f = open("output.tex", 'w')
    f.write("\documentclass{article}\n\usepackage{tikz}\n\\begin{document}\n\\begin{tikzpicture}\n")
    for node in g.nodes:
      for neighbor in node.neighbors:
        f.write("\t\\draw " + getEdgeString(node, neighbor))
    for node in g.nodes:
      f.write("\t\\draw " + getNodeString(node))   
    f.write("\end{tikzpicture}\n\end{document}")

transpile(graph)
