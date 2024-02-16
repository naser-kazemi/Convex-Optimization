import os
import numpy as np


def read_variable_columns(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split(',')
            data.append(columns)
    return data


def create_cascades(filename, num_nodes):

    # check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found")

    # read the file
    v = read_variable_columns(filename)

    # initialize C (The cascades) with -1
    C = - np.ones((len(v), num_nodes), dtype=float)

    # loop through each row in v
    for i in range(len(v)):
        if int(v[i][0]) < num_nodes:
            C[i, int(v[i][0])] = v[i][1]

        j = 2
        while j < len(v[i]) and int(v[i][j]) > -1:
            if int(v[i][j]) < num_nodes:
                C[i, int(v[i][j])] = v[i][j+1]
            j += 2

    return C


def create_adj_matrix(filename, num_nodes):
    # initialize the adjacency matrix with zero
    A = np.zeros((num_nodes, num_nodes), dtype=float)

    # check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found")

    # read the file
    v = read_variable_columns(filename)
    v = v[:num_nodes]

    # loop through each row in v to populate the adjacency matrix
    for i in range(len(v)):
        if int(v[i][0]) < num_nodes and int(v[i][1]) < num_nodes:
            A[int(v[i][0]), int(v[i][1])] = v[i][2]

    return A


if __name__ == "__main__":
    network_file = '../Data/kronecker-core-periphery-n1024-h10-r0_01-0_25-network.txt'
    cascades_file = '../Data/kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt'

    num_nodes = 200  # number of nodes in Kronecker graph

    print(create_adj_matrix(network_file, num_nodes))
    print(create_cascades(cascades_file, num_nodes))
