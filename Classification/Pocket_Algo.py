import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

alpha = 0.001
mina = 2000
minw = []
plot_x = []
plot_y = []
times = 0
ITERATIONS = 7000

def isValid(prod1,label, rows, w) :
	global mina
	global minw
	global plot_x
	global plot_y
	global times
	prod = prod1[0]
	a = []
	for i in range(0,rows,1):
		if (prod[i] >= 0) and (label[i] == 1):
			continue
		elif (prod[i] < 0) and (label[i] == -1):
			continue
		else:
			a.append(i)
	times  = times + 1
	plot_x.append(times)
	if len(a) > 0 :
		if mina > len(a) :
			mina = len(a)
			minw = w
		ind = random.randrange(0,len(a))
		plot_y.append(len(a))
		return a[ind]
	else:
		mina = 0
		minw = w
		plot_y.append(0)
	return -1

def pocket(data, label, rows):
	a = []
	global ITERATIONS
	for i in range(0,4) :
		#a.append(random.uniform(-1,1))
		a.append(4.0)
	w = np.array([a])
	count = 0
	prod = np.dot(w, data.T)
	index = isValid(prod, label, rows, w)
	while (index != -1) and count < ITERATIONS:
		term = np.dot(w, data[index].T)
		if (label[index] == 1) and prod[0][index] < 0:
			while term[0] < 0:
				w = w + alpha*(data[index])
				term = np.dot(w, data[index].T)
		else:
			while term[0] >= 0:
				w = w - alpha*(data[index])
				term = np.dot(w, data[index].T)
		count  = count + 1
		prod = np.dot(w, data.T)
		index = isValid(prod, label, rows, w)
	print "The no. of violated terms for best solution:"
	print  mina
	print "The best possible weights:"
	print minw
	print "The iterations taken: " + str(count)
	print "Accuaracy:" + str((float)(rows-mina)/rows)
	plt.plot(plot_x, plot_y)
	plt.xlabel('Iteration Number')
	plt.ylabel('Number of Misclassified points')
	plt.title('Pocket Algorithm on Unpromised data points')
	plt.show()
	return

if __name__ == '__main__':
	df = pd.read_csv('/home/ajitesj/Desktop/HW4/classification.txt', sep = ",", header = None)
	rows = df[0].count()
	df1 = df.iloc[0:rows,0:3]
	labels = df[4]
	features = df1.as_matrix()
	intercept = np.ones((features.shape[0],1))
	features = np.hstack((intercept,features))
	pocket(features, labels, rows)
