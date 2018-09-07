import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def PCA_numpy(data):
    # 1st step is to find covarience matrix
    data_vector = []
    for i in range(data.shape[1]):
        data_vector.append(data[:, i])

    cov_matrix = np.cov(data_vector)

    # 2nd step is to compute eigen vectors and eigen values
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig_values = np.reshape(eig_values, (len(cov_matrix), 1))

    print ("Eigen Values : \n")
    print(eig_values)

    print ("Eigen vectors : \n")
    print(eig_vectors)



    # Make pairs
    eig_pairs = []
    for i in range(len(eig_values)):
        eig_pairs.append([np.abs(eig_values[i]), eig_vectors[:, i]])



    eig_pairs.sort()
    eig_pairs.reverse()

    print eig_pairs

    # This PCA is only for 2 components
    reduced_data = np.hstack(
        (eig_pairs[0][1].reshape(len(eig_pairs[0][1]), 1), eig_pairs[1][1].reshape(len(eig_pairs[0][1]), 1)))



    return data.dot(reduced_data)


if __name__ == '__main__':
    df = pd.read_csv('/Users/manojrajalbandi/Desktop/Study/ML/hw3/pca-data.txt', sep = '\t', header = None)
    print("Original 3-Dimension Data\n")
    print df.values
    reduced_data_numpy = PCA_numpy(df.values)
    print("Reduced 2-Dimension Data\n")
    print reduced_data_numpy


    plt.scatter(reduced_data_numpy[:, 0], reduced_data_numpy[:, 1], label='Original points reduced')
    plt.title("PCA numpy")
    plt.legend()
    plt.show()
