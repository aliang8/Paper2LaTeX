from graph import *

def scale_graph(graph, scale_x, scale_y):
    max_y = max(node.pos[0] for node in graph.nodes)
    max_x = max(node.pos[1] for node in graph.nodes)

    for node in graph.nodes:
        node.pos = (
            min((float(scale_y)) * node.pos[0] / float(max_y), 50),
            min((float(scale_x)) * node.pos[1] / float(max_x), 50))

def transpile(g, scale_x=15, scale_y=11):
    color="red!20"
    shape="circle"
    size="1.5cm"

    def get_node_string(node):
        return "({0},{1}) node[{2}, fill={3}, minimum size={4}] {5};\n".format(node.pos[0], -1 * node.pos[1], shape, color, size, "{}")
    def get_edge_string(node1, node2):
        return "({0},{1}) -- ({2},{3});\n".format(node1.pos[0], -1 * node1.pos[1], node2.pos[0], -1 * node2.pos[1])


    scale_graph(g, scale_x, scale_y)

    with open("output.tex", 'w') as f:
        f.write("\documentclass{article}\n\usepackage{tikz}\n\\begin{document}\n\\begin{tikzpicture}\n")
        for node in g.nodes:
            for neighbor in node.neighbors:
              f.write("\t\\draw " + get_edge_string(node, neighbor))
        for node in g.nodes:
            f.write("\t\\draw " + get_node_string(node))
        f.write("\end{tikzpicture}\n\end{document}")


if __name__ == "__main__":
    node1 = Node(None, pos=(10,5))
    node2 = Node(None, pos=(12,1))
    node3 = Node(None, pos=(3,4))
    node4 = Node(None, pos=(4,9))
    node1.neighbors = set([node2, node3, node4])
    node2.neighbors = set([node1,node3])
    node3.neighbors = set([node1])
    graph = Graph({node1, node2, node3, node4})

    transpile(graph)
