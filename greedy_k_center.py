from shared_utils import *

def greedy_k_center(C,F,k, groups, alpha=None, beta=None, delta=None, epsilon=1e-3):
    if delta is None:
        if alpha is None or beta is None:
            raise Exception("alpha, beta, and delta cannot be all None")
    else:
        alpha, beta = calculate_alpha_beta(C,F,k,groups, delta)
    t = time.time()

    S_idx = greedy_helper(F,k)
    S = F[S_idx]
    d = distance_matrix(C, S)

    clusters = {i : [] for i in range(k)}
    cost = 0
    for i in range(len(C)):
        closest_center = np.argmin(d[i])
        clusters[closest_center].append(i)
        cost = max(cost, d[i,closest_center])

    time_taken = time.time()-t

    l = len(groups)
    points = {i : [] for i in range(len(C))}
    for g_i in range(l):
        for point in groups[g_i]:
            points[point].append(g_i)

    violations = max_add_violation(C, S, groups, clusters, alpha, beta, points)

    return violations, time_taken, cost

