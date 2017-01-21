import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
import cv2
try: # Compatibility with different OpenCV versions.
    import cv2.cv as cv
    HOUGH_GRADIENT = cv.CV_HOUGH_GRADIENT
except:
    HOUGH_GRADIENT = cv2.HOUGH_GRADIENT
import sys

def microstructure(l=256):
    """
    Synthetic binary data: binary microstructure with blobs.

    Parameters
    ----------

    l: int, optional
        linear size of the returned image

    """
    n = 5
    x, y = np.ogrid[0:l, 0:l]
    mask = np.zeros((l, l))
    generator = np.random.RandomState(1)
    points = l * generator.rand(2, n**2)
    mask[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
    mask = ndi.gaussian_filter(mask, sigma=l/(4.*n))
    return mask > mask.mean()

# data = microstructure(l=64)
file_name = sys.argv[1]
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
img = cv2.medianBlur(img, 5)

width, height = img.shape
while 900 < width:
    img = cv2.pyrDown(img)
    width, height = img.shape
data = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, -5)

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
ax1.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax1.axis('off')
ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
ax2.contour(data, [0.5], colors='w')
ax2.axis('off')

fig.tight_layout()
plt.show()