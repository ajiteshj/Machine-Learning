import pandas as pd
import numpy as np
from cvxopt import matrix,solvers

def read_data():
    df = pd.read_csv('data/nonlinsep.txt', header=None, names=['X', 'Y', 'label'])
    return df

def kernel_function(df):
    Z = np.array([df['X'] ** 0, df['X'] ** 2, df['Y'] ** 2,  np.sqrt(2)* df['X'], np.sqrt(2)*df['Y'], np.sqrt(2)*df['X']*df['Y']])
    return Z.T

def quadratic_solver(points, labels):
    P = matrix(np.dot(points, points.T) * np.dot(labels, labels.T))
    q = matrix(np.ones(100) * -1)
    G = matrix(np.diag(np.ones(100) * -1))
    h = matrix(np.zeros(100))
    b = matrix([0.0])
    A = matrix(labels.T, (1, 100))
    A = matrix(A, (1, 100), 'd')

    sol = solvers.qp(P, q, G, h, A, b)
    alpha = sol['x']

    #wt = 0.0
    wt = sum(alpha * labels * points)
    print("Weight",wt)
    sv = []
    for i in range(100):
        if alpha[i] > 0.01:
            b_alpha = alpha[i]
            b_points = points[i]
            sv.append(points[i])
            b_labels = labels[i]
    b = (1/b_labels) - np.dot(wt, b_points)
    print("Support vectors",sv)
    print("b",b)




def main():
    df = read_data()
    labels = np.array(df['label'])
    labels = labels.reshape((100,1))
    points = kernel_function(df)
    quadratic_solver(points, labels)

    #pass




if __name__ == '__main__':
    main()
