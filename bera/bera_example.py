import configparser
import sys

from fair_clustering import fair_clustering
from util.configutil import read_list

config_file = "config/example_config.ini"
config = configparser.ConfigParser(converters={'list': read_list})
config.read(config_file)

# Create your own entry in `example_config.ini` and change this str to run
# your own trial
config_str = "bank" if len(sys.argv) == 1 else sys.argv[1]

print("Using config_str = {}".format(config_str))

# Read variables
data_dir = config[config_str].get("data_dir")
dataset = config[config_str].get("dataset")
clustering_config_file = config[config_str].get("config_file")
num_clusters = list(map(int, config[config_str].getlist("num_clusters")))
max_points = config[config_str].getint("max_points")
violating = config["DEFAULT"].getboolean("violating")
violation = config["DEFAULT"].getfloat("violation")

deltas = config[config_str].getlist("deltas")
alphas = config[config_str].getlist("alphas")
betas = config[config_str].getlist("betas")

if deltas is not None:
    deltas = list(map(float, deltas))
else:
    alphas = list(map(float, alphas))
    betas = list(map(float, betas))
    if alphas is None or betas is None:
        print("must input alpha, beta, or delta")
        exit(0)
    elif len(alphas) != len(betas):
        print("number of alphas must match number of betas")

for n_clusters in num_clusters:
    fair_clustering(dataset, clustering_config_file, data_dir, n_clusters, deltas, max_points, violating, violation, alphas, betas)
