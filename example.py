from lp_freq_distributor import *
from greedy_k_center import *
from loaddatasets import *
from lp_bera import *
from lp_ahmadian import *

k = 4
epsilon = 0.1
delta = 0.5
C, groups, colors, name = get_reuters()

print("KFC")
fair = fair_k_clustering(C,C,k, groups, None, None, delta, epsilon)
print(fair)

print("Greedy Algorithm")
greedy = greedy_k_center(C,C,k, groups, None, None, delta, epsilon)
print(greedy)

if colors is None:
    ahmadian = (-1, -1, -1)
else:
    print("Ahmadian et al Algorithm")
    ahmadian = lp_ahmadian(C, colors, k,  max(colors)+1, alpha=delta)
    print(ahmadian)

print("Bera et al")
bera = lp_bera(k, None, None, delta, dataset=name, final_code = 'bera/')
print(bera)

print()
print('======Result======')
print("additive violation, time, cost:")
print("fair: ", fair)
print("greedy: ", greedy)
print("ahmadian: ", ahmadian)
print("bera: ", bera)
