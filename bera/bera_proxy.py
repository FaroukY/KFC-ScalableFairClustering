import sys, os
sys.path.append(os.path.dirname(__file__))

from fair_clustering import fair_clustering

def bera_proxy(dataset, n_clusters, alpha, beta):

    clustering_config_file = os.path.dirname(__file__) + '/config/dataset_configs.ini'
    data_dir =  os.path.dirname(__file__) + '/output/'
    deltas = None
    max_points = 0
    violating = False
    violation = None
    alphas = [alpha]
    betas = [beta]

    fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, deltas, max_points, violating, violation, alphas, betas)

def bera_proxy_delta(dataset, n_clusters, delta):

    clustering_config_file = os.path.dirname(__file__) + '/config/dataset_configs.ini'
    data_dir =  os.path.dirname(__file__) + '/output/'
    deltas = [delta]
    max_points = 0
    violating = False
    violation = None
    alphas = None
    betas = None

    fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, deltas, max_points, violating, violation, alphas, betas)    
    

if __name__ == "__main__":
    bera_proxy('c50', 25, 0.8, 0)