from shared_utils import *
from bera.bera_proxy import bera_proxy, bera_proxy_delta

import subprocess
from subprocess import PIPE, Popen
import os
from pathlib import Path
import json

def lp_bera(n_clusters, alpha=None, beta=None, delta=None,final_code = 'bera/', dataset='bank'):
    if delta is None:
        if alpha is None or beta is None:
            raise Exception("alpha, beta, and delta cannot be all None")

    try:
        if delta is None:
            bera_proxy(dataset, n_clusters, alpha, beta)
        else:
            bera_proxy_delta(dataset, n_clusters, delta)
    except:
        return -1, -1, -1
        
    out_dir = final_code+'/output/'
    out_file = sorted(Path(out_dir).iterdir(), key=os.path.getmtime)[-1]
    data = None
    with open(out_file) as json_file:
        data = json.load(json_file)
        
    N = len(data['points'])

    idx = 0
    subg_to_idx = {}
    points = {i : [] for i in range(N)}

    groups = {}
    alpha, beta = [], []

    for group in data['attributes'].keys():
        for subg in data['attributes'][group].keys():
            groups[idx] = data['attributes'][group][subg]
            for point in data['attributes'][group][subg]:
                points[point].append(idx)
            alpha.append(data['alpha'][group][subg])
            beta.append(data['beta'][group][subg])
            idx+=1
    alpha, beta = np.array(alpha),np.array(beta)
    
    print(alpha, beta)
    print(points)

    k = len(data['centers'])
    clusters = {i : [] for i in range(k)}
    for i in range(N):
        for j in range(k):
            if int(data['assignment'][i*k + j]) == 1:
                clusters[j].append(i)
                break


    violations = max_add_violation(None, None, groups, clusters, alpha, beta, points)
    time_taken = data['time']
    cost = data['partial_fair_score']
    return violations, time_taken, cost


def lp_bera_delta(n_clusters, delta, final_code = 'bera/', dataset='bank'):
    try:
        bera_proxy_delta(dataset, n_clusters, delta)
    except:
        return -1, -1, -1
    out_dir = final_code+'/output/'
    out_file = sorted(Path(out_dir).iterdir(), key=os.path.getmtime)[-1]
    data = None
    with open(out_file) as json_file:
        data = json.load(json_file)
        
    N = len(data['points'])

    idx = 0
    subg_to_idx = {}
    points = {i : [] for i in range(N)}

    groups = {}
    alpha, beta = [], []

    for group in data['attributes'].keys():
        for subg in data['attributes'][group].keys():
            groups[idx] = data['attributes'][group][subg]
            for point in data['attributes'][group][subg]:
                points[point].append(idx)
            alpha.append(data['alpha'][group][subg])
            beta.append(data['beta'][group][subg])
            idx+=1
    alpha, beta = np.array(alpha),np.array(beta)
    
    print(alpha, beta)
    print(points)

    k = len(data['centers'])
    clusters = {i : [] for i in range(k)}
    for i in range(N):
        for j in range(k):
            if int(data['assignment'][i*k + j]) == 1:
                clusters[j].append(i)
                break


    violations = max_add_violation(None, None, groups, clusters, alpha, beta, points)
    time_taken = data['time']
    cost = data['partial_fair_score']
    return violations, time_taken, cost
