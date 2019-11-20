import numpy as np
from utils import drawRRrobot,bruteforce_search,numeric_IK
from mpl_toolkits import mplot3d


# end effector destination position
xd = np.array([1.2, 1.2])
# initial value of theta1, theta2
q_initial = np.array([2.2, 1.8])

#stopping criterion
epsilon = 1e-3
max_iter = 100
#iteratively solve
for isNewton in [True,False]:
    q_solution, actual_iter = numeric_IK(xd, q_initial, isNewton, epsilon, max_iter)
    print('Numerical solution using','Newton\'s method' if isNewton else 'Gradient descent','is:')
    print(q_solution % (2 * np.pi))
    print('Done in ',actual_iter,'iterations.')

#brute forece solve
bruteforce_search(xd)
