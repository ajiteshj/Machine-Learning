import numpy as np

import pandas as pd


def sigmoid(scores):
    return np.exp(scores) / (1 + np.exp(scores))

def logistic_regression(features, target, num_steps, learning_rate, add_intercept=False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros((features.shape[1], 1))





    for step in xrange(num_steps):
        scores = np.dot(weights.T, features.T)
        predictions = sigmoid(scores)
        mismatch = sigmoid(-scores)




        gradient = np.dot(features.T, mismatch.T)/ 7000


        weights += learning_rate * gradient


    return weights


def main():
    df = pd.read_csv('/home/ajitesj/Desktop/HW4/classification.txt', sep = ',', header = None)

    feats = df.as_matrix(columns = [0,1,2])
    labels = df.as_matrix(columns = [4])


    weights = logistic_regression(feats, labels, num_steps= 7000, learning_rate= 0.001, add_intercept= True)

    print "Weights after Final Iteration :\n"
    print weights


    final_scores = np.dot(np.hstack((np.ones((feats.shape[0], 1)),feats)), weights)
    preds = sigmoid(final_scores)


    preds[preds < 0.5 ] = -1
    preds[preds >= 0.5] = 1



    print '\nAccuracy : {0} '.format((preds == labels).sum().astype(float)  / (len(preds)))




if __name__ == "__main__":
    main()
