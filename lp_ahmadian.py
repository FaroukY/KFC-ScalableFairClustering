from shared_utils import *
from math import log, ceil
import multiprocessing
import cplex 

class Cluster(object):
    def __init__(self, C):
        self.color_freq = [0 for _ in range(C)]
        self.tot_balls = 0
        self.variables = [[] for _ in range(C)]
        
    def add_color(self, i, v):
        self.color_freq[i]+=v
        self.tot_balls += v
    def add_variable(self, c, var):
        self.variables[c].append(var)
    def valid(self, i, alpha):
        tot_balls = self.tot_balls
        this_freq = self.color_freq[i]
        return this_freq <= alpha*tot_balls, this_freq-alpha*tot_balls
    def violation(self, i, alpha):
        valid, error = self.valid(i, alpha)
        if valid:
            return 0
        else:
            return error
    def max_violation(self, alpha):
        max_err = 0
        for i in range(len(self.color_freq)):
            max_err = max(max_err, self.violation(i, alpha))
        return max_err
    def __str__(self):
        return str(self.color_freq)
    def __rep__(self):
        return str(self.color_freq)
    def add_variable(self, color, var):
        return self.variables[color].append(var)
    def add_constraints(self, Lp_prob, alpha, beta=0):
        rhs = self.tot_balls
        sum_variables = [ p.lpSum(self.variables[c]) for c in range(len(self.variables))]
        rhs += p.lpSum(sum_variables)
        
        for c in range(len(self.color_freq)):
            Lp_prob += ( (self.color_freq[c]+sum_variables[c])<= alpha*rhs)
            Lp_prob += ( (self.color_freq[c]+sum_variables[c])>= beta*rhs)

            
def calculate_lp(_lambda, X,colors, F,k, C, alpha ):
    print("Starting calculation for lambda=",_lambda)
        
    
    t1 = time.time()
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    
    num_points = len(X)
    num_centers = len(F)

    # Name the variables -- x_j_i is set to 1 if j th pt is assigned to ith center
    variable_names = ["x_{}_{}".format(i,j) for j in range(num_points) for i in range(num_centers)]
    variables_ys = ["y_{}".format(i) for i in range(num_centers)]
    variables = variable_names + variables_ys
    total_variables = num_points * num_centers + num_centers
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]

    objective = 1

    problem.variables.add(lb=lower_bounds,
                          ub=upper_bounds,
                          names=variables)
    print("Constructed variables in %s"%(time.time()-t1))
    
    t1 = time.time()
    constraints_row, senses, rhs = [], [], []
    
    #Constraint set 1
    constraints_row.extend(
        [ [['x_{}_{}'.format(i, j) for i in range(len(F))], [1 for i in range(len(F))]] for j in range(len(X)) ]
    )
    senses.extend(
        [ 'G' for j in range(len(X))]
    )
    rhs.extend(
        [ 1 for j in range(len(X)) ]
    )
    
    #Constraint set 2
    constraints_row.extend(
        [ [['x_{}_{}'.format(i, j), 'y_{}'.format(i)], [1 , -1]] for i in range(len(F)) for j in range(len(X))]
    )
    senses.extend(
        [ 'L' for i in range(len(F)) for j in range(len(X))]
    )
    rhs.extend(
        [ 0 for i in range(len(F)) for j in range(len(X)) ]
    )
    
    #Constraint set 3
    variables_vec = [ ['x_{}_{}'.format(i, j) for j in range(len(X))] for i in range(len(F)) ]    
    alpha_vec = {
        c : [ 1-alpha if colors[j]==c else -alpha for j in range(len(X)) ] for c in range(C)
    }
    for i in range(len(F)):
        for c in range(C):
            const = [ variables_vec[i] , alpha_vec[c] ]
            constraints_row.append(const)
        
    senses.extend(
        [ 'L' for i in range(len(F)) for c in range(C) ]
    )
    rhs.extend(
        [ 0 for i in range(len(F)) for c in range(C) ]
    )
    
    
    #Constraint set 4
    ceil_alpha = ceil(1/alpha)
    ones = [1 for j in range(len(X))]
    for i in range(len(F)):
        relev_vars = variables_vec[i] + ['y_{}'.format(i)]
        coef = ones + [-ceil_alpha]
        sense = 'G'
        rh = 0
        constraints_row.append([relev_vars, coef])
        senses.append(sense)
        rhs.append(rh)
    
    #Constraint set 5
    constraints_row.append( [ ['y_{}'.format(i) for i in range(len(F))],[1 for i in range(len(F)) ]  ] )
    senses.append('L')
    rhs.append(k)
    
    #Constraint set 8
    #x[i][j] =0 if d(i,j)>_lambda
    d = distance_matrix(X[F],X)
    for i in range(len(F)):
        for j in range(len(X)):
            if d[i][j]>_lambda:
                constraints_row.append( [ ['x_{}_{}'.format(i,j)], [1] ] )
                senses.append('E')
                rhs.append(0)
    
    print("start!!!")
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs)

    
    t2 = time.time()
    print(t2-t1)
    t1 = time.time()
    
    try:
        problem.solve()
        print("Took %s to solve "%(time.time()-t1))
        
        res = {
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "objective": problem.solution.get_objective_value(),
        "assignment": problem.solution.get_values(),
        }
    except:
        return False, None, None
    
    x = np.zeros((len(F), len(X)))
    y = np.array([0 for _ in range(len(F))])
    
    for i, v in enumerate(problem.solution.get_values()):
        va = problem.variables.get_names(i)
        if va.find("x_")>=0:
            i = int(va.split("_")[1])
            j = int(va.split("_")[2])
            x[i][j] = v
        else:
            i = int(va.split("_")[1])
            y[i] = v
    
    return problem.solution.get_status(), x,y


"""
Ahmadian et al algorithm
"""
def lp_ahmadian(X, colors, k, C, alpha, epsilon=0.5):
    print(k, C, alpha)
    t = time.time()
    N = len(X)
    
    def helper(X, colors, k, C, alpha, centers_idx = None):
        if centers_idx is None:
            centers_idx = greedy_helper(X,k)

        max_dist = 0
        d = distance_matrix(X,X[centers_idx])
        for i in range(len(X)):
            j = d[i].argmin()
            max_dist = max(max_dist,d[i].min() )
        return max_dist
 
    def get_lamb():
        mm=2
        F = set(k_greedy(X,k))
        while len(F)<mm*k:
            F = F.union(set(k_greedy(X,k)))
        F = list(F)[:mm*k]
        print("F=",F)
        
        d = distance_matrix(X[F],X)
        
        l = helper(X, colors, k, C, alpha, None)
        l/=2
        r = 2*d.max()
        
        
        repeated_grid = ceil( log(r/l)/log(1+epsilon))+10 #some buffer for double errors
        lambdas = [l*(1+epsilon)**i for i in range(repeated_grid) if l*(1+epsilon)**i<=r ]
        
        print(lambdas)
        for lamb in lambdas:
            print(lamb)
            success,x,y = calculate_lp(lamb, X,colors, F,k, C, alpha)
            if success:
                return lamb, F, x, y
        return lamb, F, x, y
    
    lamb, F, x, y = get_lamb()
    d = distance_matrix(X[F],X)
    
    x_vals = x
    y_vals = y


    def get_F_prime(X, colors, F):
        indices = [F[0]]

        while True:
            repeat = False
            for i in range(len(F)):
                dist = d[i][indices].min()
                if dist > 2*lamb:
                    indices.append(F[i])
                    repeat = True
                    break
            if not repeat:
                break
        return indices

    def theta(X, colors, F_prime, F):
        thet = dict()
        for i in range(len(F)):
            if F[i] in F_prime:
                thet[i] = i
            else:
                for j in range(len(F_prime)):
                    if d[i][F_prime[j]]<2*lamb:
                        thet[i]=j
                        break

        theta_inv = dict()
        for k,v in thet.items():
            if v not in theta_inv:
                theta_inv[v] = []
            theta_inv[v].append(k)

        return thet, theta_inv




    F_prime = get_F_prime(X,colors, F)
    print(F_prime)
    
    thet, theta_inv = theta(X, colors, F_prime, F)

    y_prime = [1 if F[i] in F_prime else 0 for i in range(len(F)) ]
    x_prime = np.copy(x_vals)

    for i in range(len(F)):
        if F[i] in F_prime:
            for j in range(len(X)):
                x_prime[i][j]=0
                for i_prime in theta_inv[i]:
                    x_prime[i][j]+= x_vals[i_prime][j]
        else:
            x_prime[i] = 0
    print(x_prime)
    
    Layers = [
        ["s"],
        {j : {'LEFT': [0], 'RIGHT': [0]} for j in range(len(X))},
        {(i,c):{'LEFT': [0], 'RIGHT': [0]} for i in range(len(F_prime)) for c in range(C)},
        {i:{'LEFT': [0], 'RIGHT': [0]} for i in range(len(F_prime))},
        ["t"]
    ]

    obj = 0
    Lp_prob = p.LpProblem('Problem', p.LpMaximize)

    for j in range(len(X)):
        var = p.LpVariable("s-%s"%(j), lowBound=0, upBound=1)
        obj += var
        Layers[1][j]['LEFT'].append(var)
    Lp_prob += obj

    for j in range(len(X)):
        for i in range(len(F_prime)):
            for c in range(C):
                if colors[j]==c and x_prime[F.index(F_prime[i])][j]>0:
                    var = p.LpVariable("%s-%s-%s-%s"%(1,j,i,c), lowBound=0, upBound=1)
                    Layers[1][j]['RIGHT'].append(var)
                    Layers[2][(i,c)]['LEFT'].append(var)



    for i in range(len(F_prime)):
        for c in range(C):
            lb = floor(sum(x_prime[F.index(F_prime[i])][colors==c]))
            ub = ceil(sum(x_prime[F.index(F_prime[i])][colors==c]))
            var = p.LpVariable("%s-%s-%s-%s"%(2,i,c,i), lowBound=lb, upBound=ub)
            Layers[2][(i,c)]['RIGHT'].append(var)
            Layers[3][i]['LEFT'].append(var)

    for i in range(len(F_prime)):
        lb = floor(x_prime[F.index(F_prime[i])].sum())
        ub = ceil(x_prime[F.index(F_prime[i])].sum())
        var = p.LpVariable("%s-%s-t"%(3,i), lowBound=lb, upBound=ub)
        Layers[3][i]['RIGHT'].append(var)



    """
    Flow conservation constraint
    """
    for j in range(len(X)):
        Lp_prob += sum(Layers[1][j]['LEFT']) == sum(Layers[1][j]['RIGHT'])

    for i in range(len(F_prime)):
        for c in range(C):
            Lp_prob += sum(Layers[2][(i,c)]['LEFT'])==sum(Layers[2][(i,c)]['RIGHT'])
    for i in range(len(F_prime)):
        Lp_prob += sum(Layers[3][i]['LEFT'])==sum(Layers[3][i]['RIGHT'])

    print("Final solve")
    status = Lp_prob.solve()
    print(obj.value())
    assert abs(obj.value() - N)<EPSILON

    clusters = {
        i : Cluster(C) for i in range(len(F_prime))
    }
    for v in Lp_prob.variables():
        if v.name[0]=='1':
            #1, j, i, c
            j = int(v.name.split("_")[1])
            i = int(v.name.split("_")[2])
            c = int(v.name.split("_")[3])

            clusters[i].add_color(c, 1)
    violations = [clusters[i].max_violation(alpha) for i in range(len(F_prime))]

    max_dist = 0
    counter = 0
    for v in Lp_prob.variables():
        if v.name[0]=='1':
            #1, j, i, c
            j = int(v.name.split("_")[1])
            i = int(v.name.split("_")[2])
            c = int(v.name.split("_")[3])
            if v.value()>0:
                max_dist = max(max_dist, d[F.index(F_prime[i])][j])
                counter += 1
    return max(violations), time.time()-t, max_dist
