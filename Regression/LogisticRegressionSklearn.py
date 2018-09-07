from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

df = pd.read_csv('/home/ajitesj/Desktop/HW4/classification.txt', sep = ',', header = None)

feats = df.as_matrix(columns = [0,1,2])
labels = df.as_matrix(columns = [4])

labels = np.ravel(labels)


clf = LogisticRegression(max_iter= 7000, fit_intercept= True, C = 1e15)


clf.fit(feats, labels)

preds = clf.predict(feats)


print clf.intercept_, clf.coef_

print 'Accuracy from sk-learn: {0}'.format(clf.score(feats, labels))
