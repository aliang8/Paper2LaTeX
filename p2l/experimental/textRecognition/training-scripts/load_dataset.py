import os
import numpy as np
import re
import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT

path = os.path.join("..", "data", "uji")
train_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

X, y = [], []

width, height = 1300, 2100
num_pair = re.compile(r"\s*\d+\s+\d+\s*")

for train_file in train_files:
	with open(os.path.join(path, train_file), "r") as dataset:
		build = []
		for line in dataset:
			if ".SEGMENT CHARACTER" in line:
				y.append(line[line.find("\"") + 1:line.find("\"", line.find("\"") + 1)])
			elif ".PEN_UP" in line and build:
				img = np.zeros((height, width), np.uint8)
				for t in xrange(len(build) - 1):
					x1 = build[t]
					x2 = build[t + 1]
					cv2.line(img, x1, x2, 255)
				X.append(cv2.resize(img, (650, 1050)))
				build = []
			elif num_pair.match(line):
				coordinates = [int(x) for x in line.split()]
				build.append(tuple(coordinates))
# print y[0]
# cv2.imshow("img", X[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

