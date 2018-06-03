from __future__ import print_function
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, circle

from skimage import feature
from skimage import io
import skimage

from time import time
import logging
import scipy
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data

# number of features (pixels)
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

def plot_gallery(images, h, w, n_row=4, n_col=5):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)

        # reshape data - form images (height, width)
        img = images[i].reshape(h, w)

        # create mask covering only grey pixels with value below 80 (mask might not work with strong lighting)
        markers1 = np.zeros_like(img)
        markers1[img[:,:] < 80] = 1

        # create mask marking already selected circles (removes overlapping circles)
        markers2 = np.zeros_like(img)

        # apply canny edge detection on the image file
        edges = canny(img, sigma=3)

        # search for circles
        hough_radii = np.arange(3, 30, 1)
        hough_res = hough_circle(edges, hough_radii)
        # get x, y, r value of each circle
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                           total_num_peaks=8)

                                          
        for center_y, center_x, radius in zip(cy, cx, radii):
            # get array of pixels covered by the circles with half radius, on the masked image
            rr, cc = circle(center_y, center_x, radius/2, markers1.shape)
            # remove the circles located below 2/5 height of the image
            if np.any(rr<h*2/5):
                # if covered pixels grey values is below 80 (marker1)
                if np.any(markers1[rr, cc] == 1):
                    # get array of pixels covered by new circle
                    yy, xx = circle(center_y, center_x, radius, markers2.shape)
                    # if circle is NOT overlapping any previously markes circles
                    if np.all(markers2[yy, xx] == 0):
                        # include the circle covered pixels to the mask
                        yy, xx = circle(center_y, center_x, radius, markers2.shape)
                        markers2[yy, xx] = 1
                        # mark circle
                        circy, circx = circle_perimeter(center_y, center_x, radius)
                        img[circy, circx] = (255)
                        
        plt.imshow(img, cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
start = time()
plot_gallery(X, h, w)
print("Eye detection time = {:.2f} seconds".format(time() - start))
plt.show()
