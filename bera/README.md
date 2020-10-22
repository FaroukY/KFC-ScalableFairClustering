# FairClustering

Implementation of Common Fair Clustering Algorithms

There are 4 main functions/APIs to be used from the files here. Try running `python example.py`. See example.py for an example on how to use all 4 function. 

## Preliminaries 
Make sure path_to_cplex = r'/Applications/CPLEX_Studio1210/cplex/bin/x86-64_osx/cplex' in shared_utils.py is pointing to the correct cplex binary. 

## Function 1: fair_k_clustering
Inputs: C, F, groups, alpha=None, beta=None, delta=None, epsilon=1e-3
Description of Inputs:
  C: A 2D numerical numpy array representing the coordinates of the clients, or the points in the input. Nxd points.  
  
  F: A 2D numerical numpy array representing the coordinates of the facilities, or the points that can be opened as centers.
  
  You can pass C here to indicate that facilities should. be opened in client positions. 
  groups: A dictionary with l keys. Its keys must be 0,1,2,...,l-1. This represents the l groups (e.g Male, Female, Asian, Hispanic, White, ...). For each i \in {0,1,...,l-1}, groups[i] is a list with the index of the points in C that belong to this group. 
  
      For example, suppose you have l=2 (Male and Female). Also, suppose you have 4 points in C, the first 2 are males, and the second 2 are females. Then the groups dictionary would be:
        groups = {
                    0: [0,1],
                    1: [2,3]
                  }
                  
  k : number of clusters
  
  alpha: (optional) An array of length l that represents alpha[i] for group i. 
  
  beta:  (optional) An array of length l that represents beta[i] for group i. 
  
  delta: (optional) A number to calculate alpha and beta using the method from Bera et al. (Neurips 2019).
  
  NOTE: either provide (alpha AND beta) OR (delta). If you don't provide any of them, the method will throw. 
  
  epsilon: A small number to stop binary search once r-l<epsilon
  
Return:
The function returns three numeric values as a tuple. The first is the maximum additive violation of the cluster by the algorithm. The second is the time_taken for the algorithm. The third is the cost (i.e kcenter distance) of the algorithm. 

## Function 2: greedy_k_center
Inputs: C,F,k, groups, alpha, beta, delta, epsilon

For description of inputs, view above. See sample usage in example.py

Return:
The function returns three numeric values as a tuple. The first is the maximum additive violation of the cluster by the algorithm. The second is the time_taken for the algorithm. The third is the cost (i.e kcenter distance) of the algorithm. 


## Function 3: lp_ahmadian
Inputs: X,colors, k, cols , alpha

Input Description:

  X: A 2D numerical numpy array representing the coordinates of the clients, or the points in the input. Nxd points. 
  
  colors: 1D array of length len(X). colors[i] describes the color of ball i. 
  
  k: Number of clusters
  
  cols: Number of distinct colors in the colors array. 
  
  alpha: The alpha cap from Ahmadian et al Paper
  
Return:
The function returns three numeric values as a tuple. The first is the maximum additive violation of the cluster by the algorithm. The second is the time_taken for the algorithm. The third is the cost (i.e kcenter distance) of the algorithm. 

## Function 4: lp_bera
The function uses the implementation of Bera et al (NeurIPS 2019, https://github.com/nicolasjulioflores/fair_algorithms_for_clustering) for their implementation. It is simply a wrapper to the original bera code, and then parses the output in bera/output/ of their algorithm to collect comparative statistics. 

To run an experiment on a dataset on Bera et al's algorithm, you will need to do a few things:

1) Change bera/config/dataset_configs.ini and fill in information about your dataset. Refer to https://github.com/nicolasjulioflores/fair_algorithms_for_clustering for more documentation. 
2) Change bera/config/example_config.ini to add a sample test you want to try (say [bank] and its details). 
3) Call the function lp_bera() [Like in example.py]. Pass in the FinalCode directory location, as well as which test to run (for example 'bank').

Return:
The function returns three numeric values as a tuple. The first is the maximum additive violation of the cluster by the algorithm. The second is the time_taken for the algorithm. The third is the cost (i.e kcenter distance) of the algorithm. 


## License 
MIT. 