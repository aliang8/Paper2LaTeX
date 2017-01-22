import sys
from imgrec import get_graph
from genlatex import transpile

if __name__ == "__main__":
	img_name = sys.argv[1]
	transpile(get_graph(img_name))

