import cvxopt
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

samples = 100
x = np.random.uniform(0, 1, (samples,2))

d = np.empty((samples), int)

for i in range(samples):
    d[i] = 1 if x[i][1] < (1/5)*np.sin(10*x[i][0])+0.3 or (x[i][1]-0.8)**2 + (x[i][0]-0.5)**2 < (0.15)**2 else -1




