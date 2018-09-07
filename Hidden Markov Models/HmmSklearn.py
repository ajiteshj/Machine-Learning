# Using Viterbi Algorithm
import numpy as np
from hmmlearn import hmm


def int_to_coord(n):
    return [n / 10, n % 10]


def coord_to_int(c):
    return (c[0] * 10 + c[1])


# the number of possible state values
K = 100
matrix = []
freep = 0
obs = []
ROW = 10
COL = 10
start_prob = []
NO_TOWERS = 4
gtower = [[0, 0], [0, 9], [9, 0], [9, 9]]
emip = []


def free_space(i, j):
    count = 0
    if i - 1 >= 0:
        if matrix[i - 1, j] == 1:
            count += 1
    if i + 1 <= 9:
        if matrix[i + 1, j] == 1:
            count += 1
    if j - 1 >= 0:
        if matrix[i, j - 1] == 1:
            count += 1
    if j + 1 <= 9:
        if matrix[i, j + 1] == 1:
            count += 1
    return count


def trans_prob(x1, y1, x2, y2):
    #print matrix[x2,y2]
    if matrix[x2, y2] == 0:
        return 0
    if x1 == x2 and y1 == y2:
        return 0
    if x1 != x2 and y1 != y2:
        return 0
    if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
        return 0
    freespaces = free_space(x1, y1)
    return float(1) / freespaces


def cal_freep():
    count = 0
    for i in range(0, ROW):
        for j in range(0, COL):
            if (matrix[i][j] == 1):
                count = count + 1
    return count


def pie_init():
    pi = []
    global freep
    global matrix
    for i in range(0, ROW):
        for j in range(0, COL):
            if matrix[i][j] == 1:
                pi.append(float(1) / freep)
            else:
                pi.append(0)
    return pi


def emission_prob(tower, pos):
    d = np.linalg.norm(np.array(tower) - np.array(pos))
    if d == 0:
        return float(1)
    return float(1) / (d * 6)


def cal_emip():
    entry = 1
    emi = []
    for i in range(0, ROW):
        for j in range(0, COL):
            entry = emission_prob((0, 0), (i, j))
            entry = entry * emission_prob((0, 9), (i, j))
            entry = entry * emission_prob((9, 0), (i, j))
            entry = entry * emission_prob((9, 9), (i, j))
            emi.append(entry)
    return emi



def gen_transmat():
    transmat = []
    for i in range(0,100):
        temp = []
        [x1,y1] = int_to_coord(i)
        for j in range(0,100):
            [x2,y2] = int_to_coord(j)
            temp.append(trans_prob(x1,y1,x2,y2))
        transmat.append(temp)
    return transmat

def main():
    global matrix
    global obs
    global start_prob
    global emip
    matrix = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # print matrix
    obs = np.array(
        [[6.3, 5.9, 5.5, 6.7], [5.6, 7.2, 4.4, 6.8], [7.6, 9.4, 4.3, 5.4], [9.5, 10.0, 3.7, 6.6], [6.0, 10.7, 2.8, 5.8],
         [9.3, 10.2, 2.6, 5.4], [8.0, 13.1, 1.9, 9.4], [6.4, 8.2, 3.9, 8.8], [5.0, 10.3, 3.6, 7.2],
         [3.8, 9.8, 4.4, 8.8], [3.3, 7.6, 4.3, 8.5]])
    global freep
    freep = cal_freep()
    start_prob = pie_init()
    emip = cal_emip()

    transprob =  gen_transmat()
    n_states = 100
    state_space = list(range(100))


    #Create a HMM
    model = hmm.GaussianHMM(n_components=3, covariance_type="full")
    model.startprob_ = start_prob
    model.transmat_ = np.array(transprob)

    X,Z = model.sample(100)

    # Train HMM
    remodel = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
    remodel.fit(X)

    Z2 = remodel.predict(X)
    print Z2

    if __name__ == '__main__':
        main()
