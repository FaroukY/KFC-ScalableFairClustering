import pulp as p
from itertools import chain, combinations
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np
np.random.seed(0)

from scipy.spatial import distance_matrix
from math import floor, ceil
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from os import scandir, path
import time
import pandas as pd
import ast

path_to_cplex = r'/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex'
solver_cmd = p.CPLEX_CMD(path=path_to_cplex, msg=0)

solver = p.CPLEX_PY(msg=0)
EPSILON = 0.1

"""
X: Numpy array which is Nxd. N points, each with d features.
k: Number of centers to return
Returns: *Index* of k centers using greedy algorithm
"""
def greedy_helper(X, k):
    i = np.random.randint(0, len(X))
    k_centers = [i]
    while len(k_centers)<k:
        max_dist = -1
        best_center = None
        d = distance_matrix(X,X[k_centers])
        for i in range(len(X)):
            di = d[i].min()
            if di>max_dist:
                max_dist = di
                best_center = i
        k_centers.append(best_center)
    return k_centers


def max_add_violation(C, S, groups, clusters, alpha, beta, points):
    "Calculate the maximum additive violation"
    l = len(groups)
    max_additive_violation = 0
    
    #for each cluster
    for j in range(len(clusters)):
        balls = np.array(clusters[j])
        clust_size = len(balls)
        #for each group
        for a in range(l):
            relevant = 0
            #for each point in the cluster
            for point in balls:
                if a in points[point]:
                    relevant += 1
            
            #should have beta[a]*clust_size <= relevant <= alpha[a]*clust_size
            if relevant > alpha[a]*clust_size:
                max_additive_violation = max(max_additive_violation, ceil(relevant - alpha[a]*clust_size ))
            elif relevant < beta[a]*clust_size:
                max_additive_violation = max(max_additive_violation, ceil(beta[a]*clust_size - relevant))
    return max_additive_violation

def calculate_alpha_beta(C,F,k,groups, delta):
    l = len(groups.keys())
    N = len(C)
    alpha, beta = np.zeros(l), np.zeros(l)
    for i in range(l):
        ri = len(groups[i])/N
        beta[i] = ri * (1-delta)
        alpha[i] = ri / (1-delta)
    return (alpha, beta)

def k_greedy(X, k):
    i = np.random.randint(0, len(X))
    k_centers = [i]
    #d = distance_matrix(X,X)
    while len(k_centers)<k:
        max_dist = -1
        best_center = None
        d = distance_matrix(X,X[k_centers])
        for i in range(len(X)):
            di = d[i].min()
            if di>max_dist:
                max_dist = di
                best_center = i
        k_centers.append(best_center)
    return k_centers
