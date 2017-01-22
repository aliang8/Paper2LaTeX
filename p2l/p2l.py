import sys
from imgrec import get_graph
from genlatex import transpile

if __name__ == "__main__":
	img_name = sys.argv[1]
	scale_x = 15
	scale_y = 11
	if len(sys.argv) >= 3:
		scale_x = int(sys.argv[2])
	if len(sys.argv) >= 4:
		scale_y = int(sys.argv[3])
	transpile(get_graph(img_name), scale_x=scale_x, scale_y=scale_y)