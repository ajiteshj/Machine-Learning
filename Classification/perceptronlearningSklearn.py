import pandas as pd
import numpy as np
import random
from sklearn.linear_model import perceptron


# reading data from csv
def read_data():
    df = pd.read_csv("/home/ajitesj/Desktop/HW4/classification.txt", header=None)
    feats = df.as_matrix(columns = [0,1,2])
    labels = df.as_matrix(columns = [3])

    labels = np.ravel(labels)


    return feats, labels


def main():
    X,Y = read_data()
    net = perceptron.Perceptron(n_iter=7000, shuffle= False, fit_intercept=True, eta0=0.002)
    net.fit(X, Y)
    print "Prediction " + str(net.predict(X))
    print "Accuracy   " + str(net.score(X, Y) * 100) + "%"



if __name__ == '__main__':
    main()
