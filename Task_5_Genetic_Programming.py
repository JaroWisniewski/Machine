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
from sklearn.decomposition import RandomizedPCA, PCA
from scipy.stats import sem

print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


###############################################################################
# Download the data, if not already on disk and load it as numpy arrays
# Download person data with pictures number higher than 200 - Bush 530, Powell 236
lfw_people = fetch_lfw_people(min_faces_per_person=200, color = False, resize = 0.5)

# take first 200 pictures of each person 
mask = np.zeros(lfw_people.target.shape, dtype = bool)
for target in np.unique(lfw_people.target):
    mask[np.where(lfw_people.target == target)[0][:200]] = 1

X = lfw_people.data[mask]

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target[mask]
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 10

print("Extracting the top %d eigenfaces from %d faces"
      % (n_components, X.shape[0]))
pca = PCA(n_components =  n_components).fit(X)

eigenfaces = pca.components_.reshape((n_components, h, w))

print("Projecting the input data on the eigenfaces orthonormal basis")
X_pca = pca.transform(X)

print("Number of pictures : {}".format(len(X)))
print("Bush pictures = {}".format(sum(y == 1)))


###############################################################################
#GP
# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 10)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.tanh, 1)
#pset.addPrimitive(math.acosh, 1)
#pset.addPrimitive(math.sqrt, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-10,10))
pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
pset.renameArguments(ARG3='d')
pset.renameArguments(ARG4='e')
pset.renameArguments(ARG5='f')
pset.renameArguments(ARG6='g')
pset.renameArguments(ARG7='h')
pset.renameArguments(ARG8='i')
pset.renameArguments(ARG9='j')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
"""
expr = toolbox.individual()
nodes, edges, labels = gp.graph(expr)

import pygraphviz as pgv

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.pdf")
"""

# 0 = powell, 1 = bush
def evalSymbReg(individual, ):
    
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    fitness = 0
    for i in range(len(X_pca)):
        x = X_pca[i]
        value = func(*x)
        lbl = y[i]
        if value < 0 and lbl == 0:
            fitness += 1
        elif value > 0 and lbl == 1:
            fitness += 1
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.3, 1000, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof[0])    
    return pop, log, hof
    
if __name__ == "__main__":
    main()
