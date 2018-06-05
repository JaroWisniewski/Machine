from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA, PCA
from scipy.stats import sem
from sklearn.cluster import DBSCAN, KMeans
from skimage import exposure
from skimage.feature import canny
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score,completeness_score,v_measure_score

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
#CHange the dataset to only 2 people : Bush & Powell
lfw_people = fetch_lfw_people(min_faces_per_person=150, color = False, resize = 0.5)

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

###############################################################################
# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25)


###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 10#20

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components =  n_components).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))


print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

print(len(X_train_pca))

###############################################################################
#GP
# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand1110111113", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# 0 is powell, 1 is bush
def evalSymbReg(individual):
    fitness = 0
    
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    for i in range(len(X_train_pca)):
        x = X_train_pca[i]
        value = func(x)
        value2 = y_train[i]
        if (value2 == 0  and value[i] > 0) or (value2 ==1 and value[i] < 0):
            fitness += 1
        else:
            fitness -= 1
    return fitness,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    # print log
    return pop, log, hof
#first individual hof[0] is the best individual
    # graphics - print the tree

if __name__ == "__main__":
    main()


















################################################################################
## K-MEANS clustering using the n_classes attribute which gives the # of different people
#
#t0 = time()
#
#kmeans = KMeans(n_clusters=n_classes, random_state=42).fit(X_train, y_train)
#kmeans.labels_
#
################################################################################
## Quantitative evaluation of the model quality on the test set
#
#print("Predicting people's names on the test set")
#t0 = time()
#y_pred = kmeans.fit_predict(X_test, y_test)
#print("done in %0.3fs" % (time() - t0))
#print("Unique labels: {}".format(np.unique(y_pred)))
#
#print("Number of points per cluster = {}".format(np.bincount(y_pred+1)))
#print(classification_report(y_test, y_pred, target_names=target_names))
#print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
#print("HOMO SCORE")
#homo = homogeneity_score(y_test, y_pred)
#print(homo)
#print("COMPLETE")
#complete = completeness_score(y_test, y_pred)
#print(complete)
#print("FINAL")
#complete = v_measure_score(y_test, y_pred)
#print(complete)
#plt.imshow(X_test[0].reshape(h,w))
#plt.show()

