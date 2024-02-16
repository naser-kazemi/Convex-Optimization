import os
import pickle

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from netrate import netrate
from generate_data import create_adj_matrix, create_cascades


network_file = '../Data/kronecker-core-periphery-n1024-h10-r0_01-0_25-network.txt'
cascades_file = '../Data/kronecker-core-periphery-n1024-h10-r0_01-0_25-1000-cascades.txt'


horizon = 20
type_diffusion = 'exp'
num_nodes = 200  # number of nodes in Kronecker graph


A_hat, total_obj, pr, mae = netrate(
    network_file, cascades_file, horizon, type_diffusion, num_nodes)

# convert A_hat to matrix
A_hat = A_hat.toarray()

A = create_adj_matrix(network_file, num_nodes)

# print the results
print(f'Precision: {pr[0]}')
print(f'Recall: {pr[1]}')
print(f'Mean Absolute Error: {mae}')
print(f'Total Objective: {total_obj}')

# Save the results
network = network_file.split('/')[-1].split('.')[0]

# make the Result/{network} directory if it does not exist
if not os.path.exists(f'Result/{network}'):
    os.mkdir(f'Result/{network}')


with open(f'solution.pkl', 'wb') as f:
    pickle.dump({'A_hat': A_hat, 'mae': mae,
                'pr': pr, 'total_obj': total_obj}, f)


# save the plots, beautify the plots with seaborn for the graphs. note that the graphs are relatively large
G = nx.from_numpy_array(A_hat)
nx.draw(G, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue', font_size=8, edge_color='gray',
        linewidths=0.5, width=0.5, alpha=0.7, pos=nx.spring_layout(G), arrowsize=10, arrowstyle='->', connectionstyle='arc3, rad = 0.1', edge_cmap=plt.cm.Blues)
plt.savefig(f'Result/{network}/A_hat_{num_nodes}.png', dpi=300)

G = nx.from_numpy_array(A)
nx.draw(G, with_labels=True, font_weight='bold', node_size=100, node_color='skyblue', font_size=8, edge_color='gray',
        linewidths=0.5, width=0.5, alpha=0.7, pos=nx.spring_layout(G), arrowsize=10, arrowstyle='->', connectionstyle='arc3, rad = 0.1', edge_cmap=plt.cm.Blues)
plt.savefig(f'Result/{network}/A_{num_nodes}.png', dpi=300)
