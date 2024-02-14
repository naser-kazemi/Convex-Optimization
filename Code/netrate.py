import numpy as np
import os
import pickle


from generate_data import create_adj_matrix, create_cascades
from estimate_network import estimate_network


def netrate(network, cascades, horizon, type_diffusion, num_nodes):
    min_tol = 1e-4

    pr = np.zeros(2)

    print('Reading ground-truth...')
    A = create_adj_matrix(network, num_nodes)

    print('Reading cascades...')
    C = create_cascades(cascades, num_nodes)

    print('Building data structures...')
    A_hat, total_obj = estimate_network(
        A, C, num_nodes, horizon, type_diffusion)

    if os.path.exists(network):
        non_zero_mask = (A != 0)
        A_non_zero = A[non_zero_mask]
        A_hat_non_zero = A_hat[non_zero_mask]
        mae = np.mean(np.abs(A_hat_non_zero - A_non_zero) / A_non_zero)

        # Compute precision and recall
        A_hat_binary = A_hat > min_tol
        A_binary = A > min_tol

        # Element-wise logical AND using the multiply method
        intersection = A_hat_binary.multiply(A_binary)
        recall = intersection.sum() / A_binary.sum()
        precision = intersection.sum() / A_hat_binary.sum()
        pr = np.array([precision, recall])
    else:
        mae = None
        pr = None

    network = network.split('/')[-1].split('.')[0]

    # Saving the results
    with open(f'Result/solution-{network}.pkl', 'wb') as f:
        pickle.dump({'A_hat': A_hat, 'mae': mae,
                    'pr': pr, 'total_obj': total_obj}, f)

    return A_hat, total_obj, pr, mae
