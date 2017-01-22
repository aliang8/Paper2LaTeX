import os
import re

path = os.path.join("..", "data", "uji")
train_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

X, y = [], []

num_pair = re.compile(r"\s*\d+\s+\d+\s*")

for train_file in train_files:
	with open(os.path.join(path, train_file), "r") as dataset:
		build = []
		for line in dataset:
			if ".SEGMENT CHARACTER" in line:
				y.append(line[line.find("\"") + 1:line.find("\"", line.find("\"") + 1)])
			elif ".PEN_DOWN" in line and build:
				X.append(build)
			elif num_pair.match(line):
				build.append(tuple(line.split()))

print y