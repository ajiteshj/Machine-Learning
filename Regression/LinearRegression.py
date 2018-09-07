import numpy  as np

Data_list = []
Y = []


def ReadFile():
    list_of_coordinates = []
    X_list, Y_list = [], []
    global Data_list
    global Y
    with open("/home/ajitesj/Desktop/HW4/linear-regression.txt", "r") as fo:
        for line in fo:
            list_of_coordinates.append(line)
    fo.close()
    for line in list_of_coordinates:
        list_of_items_in_line = line.split(",")
        X_list.append(float(list_of_items_in_line[0]))
        Y_list.append(float(list_of_items_in_line[1]))
        Y.append([float(list_of_items_in_line[2])])
    Data_list.append([1] * len(X_list))
    Data_list.append(X_list)
    Data_list.append(Y_list)


def LinearReg():
    global Data_list, Y
    Data_list = np.matrix(Data_list)
    Dtranspose = Data_list.T
    W = np.linalg.inv((Data_list * Dtranspose))
    P1 = Data_list * Y


    print"---------------------Output-------------------------"
    Wout = W * P1
    Wout = np.matrix(Wout)
    print " Final Weights :\n [W0           W1           W2    ]\n", Wout.T

def main():
    ReadFile()
    LinearReg()



if __name__ == "__main__":
    main()
