# -*- coding: utf-8 -*-
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import scipy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.svm import SVC
from scipy.stats import sem
from skimage.feature import peak_local_max, canny
from skimage import exposure

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays

lfw_people = fetch_lfw_people(min_faces_per_person=70)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)

X = lfw_people.data  
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print(h)
print(w)


###############################################################################
# Split into a training set and a test set using a stratified k fold

# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)

"""
###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 100

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, whiten=True, random_state=0).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


###############################################################################
# Train a SVM classification model
"""
print("Fitting the classifier to the training set")

from sklearn.cluster import DBSCAN, KMeans
kmeans = KMeans(n_clusters=n_classes, random_state=0, n_init=30).fit(X_train, y_train)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = kmeans.fit_predict(X_test, y_test)
print("done in %0.3fs" % (time() - t0))
print("Unique labels: {}".format(np.unique(y_pred)))

print("Number of points per cluster = {}".format(np.bincount(y_pred+1)))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


###############################################################################
# Qualitative evaluation of the predictions using matplotlib
for cluster in range(max(y_pred) +1):
     mask = y_pred == cluster
     n_images = np.sum(mask)
     fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks':(), 'yticks':()})
     figa = plt.gcf()
     figa.canvas.set_window_title("Clustering - {}".format(cluster))
     for images, label, ax in zip(X_test[mask], y_test[mask], axes):
         ax.imshow(images.reshape((h, w)), cmap=plt.cm.gray)
         ax.set_title(lfw_people.target_names[label].split()[-1])

Z = lfw_people.data

for i in range(n_samples):
    image = Z[i].reshape(h, w)
    edges = canny(image, sigma=3)
    Z[i] = edges.reshape(h*w)

X_train, X_test, y_train, y_test = train_test_split(
    Z, y, test_size=0.25)
    
kmeans = KMeans(n_clusters=n_classes, random_state=0, n_init=30).fit(X_train, y_train)


###############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = kmeans.fit_predict(X_test, y_test)
print("done in %0.3fs" % (time() - t0))
print("Unique labels: {}".format(np.unique(y_pred)))

print("Number of points per cluster = {}".format(np.bincount(y_pred+1)))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
        
for cluster in range(max(y_pred) +1):
    mask = y_pred == cluster
    n_images = np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks':(), 'yticks':()})
    figa = plt.gcf()
    figa.canvas.set_window_title("Clustering with edge detection - {}".format(cluster))
    for images, label, ax in zip(X_test[mask], y_test[mask], axes):
         ax.imshow(images.reshape((h, w)), cmap=plt.cm.gray)
         ax.set_title(lfw_people.target_names[label].split()[-1])

         
plt.show()
