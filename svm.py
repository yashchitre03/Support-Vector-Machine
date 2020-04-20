import cvxopt
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

n = 100
alpha_threshold = 1e-5
x = np.random.uniform(0, 1, (n,2))
d = np.ones((n))
C1 = []
C_minus1 = []

for i in range(n):
    if x[i,1] < (1/5)*np.sin(10*x[i,0])+0.3 or (x[i,1]-0.8)**2 + (x[i,0]-0.5)**2 < (0.15)**2:
        C1.append(x[i])
    else:
        d[i] = -1
        C_minus1.append(x[i])
        
plt.plot(*zip(*C1), '^', c='r', label='Class 1')
plt.plot(*zip(*C_minus1), 's', c='b', label='Class -1')
plt.xlabel("X Co-Ordiantes")
plt.ylabel("Y Co-Ordinates")
plt.axis((0,1,0,1))
plt.legend()
plt.show()

linear_kernel = lambda xi, xj : np.dot(xi,xj)
polynomial_kernel = lambda xi, xj, dim=5 : (1 + np.dot(xi, xj)) ** dim
gaussian_kernel = lambda xi, xj, sigma=5 : np.exp(-np.linalg.norm(xi-xj)**2 / (2 * (sigma ** 2)))

K = np.empty((n, n))
for i in range(n):
    for j in range(n):
        K[i,j] = polynomial_kernel(x[i], x[j])
            
P = cvxopt.matrix(np.outer(d,d) * K)
q = cvxopt.matrix(np.ones(n) * -1)
G = cvxopt.matrix(np.eye(n) * -1)
h = cvxopt.matrix(np.zeros(n))
A = cvxopt.matrix(d, (1,n))
b = cvxopt.matrix(0.0)

qp_solution = cvxopt.solvers.qp(P, q, G, h, A, b)

alpha = np.ravel(qp_solution['x'])
alpha_indices = np.argwhere(alpha>alpha_threshold).flatten()

alpha_sv = alpha[alpha_indices]
x_sv = x[alpha_indices]
d_sv = d[alpha_indices]

theta = d_sv[0] - sum([alpha_sv[i] * d_sv[i] * polynomial_kernel(x_sv[i], x_sv[0]) for i in range(len(x_sv))])

detail = 100
x_coordinates = np.linspace(0.0, 1.0, num=detail)
y_coordinates = np.linspace(0.0, 1.0, num=detail)

height = np.zeros((detail, detail))
for i in range(detail):
    for j in range(detail):
        height[j][i] = sum([alpha_sv[k] * d_sv[k] * polynomial_kernel(x_sv[k], np.array([x_coordinates[i], y_coordinates[j]])) for k in range(len(x_sv))]) + theta
    print(i)     
        
plt.plot(*zip(*C1), 'x', c='r', label='Class 1')
plt.plot(*zip(*C_minus1), '.', c='b', label='Class -1')
plt.xlabel("X Co-Ordiantes")
plt.ylabel("Y Co-Ordinates")
plt.contour(x_coordinates, y_coordinates, height, levels=[-1, 0, 1], colors=['b','g','r'])
plt.legend()
plt.show()

