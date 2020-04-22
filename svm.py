import cvxopt
import numpy as np
import matplotlib.pyplot as plt

# seed set to reproduce the results for report
np.random.seed(25)

n = 100                         # number of training samples
alpha_threshold = 1e-5          # Used to set alpha to 0 if very small
x = np.random.uniform(0, 1, (n,2))
d = np.ones((n))
C1 = []
C_minus1 = []

# desired outputs
for i in range(n):
    if x[i,1] < (1/5)*np.sin(10*x[i,0])+0.3 or (x[i,1]-0.8)**2 + (x[i,0]-0.5)**2 < (0.15)**2:
        C1.append(x[i])
    else:
        d[i] = -1
        C_minus1.append(x[i])
   
# input plot with desired outputs
fig, ax = plt.subplots(figsize=(10,10))
plt.plot(*zip(*C1), '^', c='r', markersize=5, label='Class 1')
plt.plot(*zip(*C_minus1), 's', c='b', markersize=5, label='Class -1')
plt.plot(np.linspace(0, 1, 1000), [0.2 * np.sin(10 * x) + 0.3 for x in np.linspace(0, 1, 1000)], linestyle="--", color="k")
plt.plot(np.linspace(0, 1, 1000), [np.sqrt(0.15 ** 2 - (x - 0.5) ** 2) + 0.8 for x in np.linspace(0, 1, 1000)], linestyle="--", color="k")
plt.plot(np.linspace(0, 1, 1000), [-np.sqrt(0.15 ** 2 - (x - 0.5) ** 2) + 0.8 for x in np.linspace(0, 1, 1000)], linestyle="--", color="k")
plt.title("Data with desired outputs and regions")
plt.xlabel("X Coordiantes")
plt.ylabel("Y Coordinates")
plt.axis((-0.01,1.01,-0.01,1.01))
plt.legend(loc=(1.01,0.5))
plt.show()

# different kernels defined
linear_kernel = lambda xi, xj : np.dot(xi,xj)
polynomial_kernel = lambda xi, xj, dim=15 : (1 + np.dot(xi, xj)) ** dim
gaussian_kernel = lambda xi, xj, sigma=2 : np.exp(-np.linalg.norm(xi-xj)**2 / (2 * (sigma ** 2)))

# calcualte Kernel K
K = np.empty((n, n))
for i in range(n):
    for j in range(n):
        K[i,j] = polynomial_kernel(x[i], x[j])
  
# define all parameters and call library to solve QP          
P = cvxopt.matrix(np.outer(d,d) * K)
q = cvxopt.matrix(np.ones(n) * -1)
G = cvxopt.matrix(np.eye(n) * -1)
h = cvxopt.matrix(np.zeros(n))
A = cvxopt.matrix(d, (1,n))
b = cvxopt.matrix(0.0)

qp_solution = cvxopt.solvers.qp(P, q, G, h, A, b)

# get alpha values
alpha = np.ravel(qp_solution['x'])
alpha_indices = np.argwhere(alpha>alpha_threshold).flatten()

# filter all support vectors
alpha_sv = alpha[alpha_indices]
x_sv = x[alpha_indices]
d_sv = d[alpha_indices]

# calulate theta
theta = d_sv[0] - sum([alpha_sv[i] * d_sv[i] * polynomial_kernel(x_sv[i], x_sv[0]) for i in range(len(x_sv))])

detail = 1000
x_coordinates = np.linspace(0.0, 1.0, num=detail)
y_coordinates = np.linspace(0.0, 1.0, num=detail)

# calculate the descriminant 
height = np.zeros((detail, detail))
for i in range(detail):
    for j in range(detail):
        height[j][i] = sum([alpha_sv[k] * d_sv[k] * polynomial_kernel(x_sv[k], np.array([x_coordinates[i], y_coordinates[j]])) for k in range(len(x_sv))]) + theta
        
# plot final support vectors and separating hyperplane
fig, ax = plt.subplots(figsize=(10,10))
plt.plot(*zip(*C1), '^', c='r', markersize=5, label='Class 1')
plt.plot(*zip(*C_minus1), 's', c='b', markersize=5, label='Class -1')
plt.plot(*zip(*x_sv), 'o', c='black', fillstyle='none', markersize=15, label='Support Vectors')
plt.axis((-0.01,1.01,-0.01,1.01))
plt.title("Using Kernel SVM")
plt.xlabel("X Coordiantes")
plt.ylabel("Y Coordinates")
contours = plt.contour(x_coordinates, y_coordinates, height, levels=[-1, 0, 1], colors=['b','g','r'])
plt.clabel(contours)
plt.legend(loc=(1.01,0.5))
plt.show()

