# List of conferences in 4area
1. Database
* SIGMOD, VLDB, ICDE, PODS, EDBT
2. Data Mining
* KDD, ICDM, SDM, PKDD, PAKDD
3. Information Retrieval
* SIGIR, ECIR, WSDM, WWW, CIKM
4. Machine Learning
* ICML, ECML, IJCAI, AAAI, CVPR

Publication detail of the authors in the above 4 areas were extracted from dblp on 1 May 2020 and stored as .json in the folder 4area/, as the dataset in Ahmadian et al. was not available upon request. We then marked the group the authors according to the main category of conference(s) they published to, with that we created a co-authorship graph using networkx, and exported an adjacency list called dblp.adjlist. Lastly, we fed the list into DeepWalk to get dblp.embeddings using the default Deepwalk settings, where each author is represented as a 8-dim vector.