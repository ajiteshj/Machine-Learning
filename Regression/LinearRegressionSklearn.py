from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np


df = pd.read_csv('/home/ajitesj/Desktop/HW4/linear-regression.txt', sep = ',', header = None)
feats = df.as_matrix(columns = [0,1])
labels = df.as_matrix(columns = [2])



clf = LinearRegression()

clf.fit(feats, labels)

preds = clf.predict(feats)

print clf.intercept_, clf.coef_
print 'Accuracy from sk-learn: {0}'.format(clf.score(feats, labels))
