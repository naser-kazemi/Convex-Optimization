import numpy as np
from scipy.sparse import lil_matrix
import cvxpy as cp


def add_log_survival(A, i, j, diff, type_diffusion):
    if type_diffusion == 'exp':
        A[i, j] += diff
    elif type_diffusion == 'pl':
        A[i, j] += np.log(diff)
    elif type_diffusion == 'rayleigh':
        A[i, j] += 0.5 * (diff) ** 2


def add_constraints(constraints, a_hat, t_hat, c_act, val, idx_ord, cidx, type_diffusion):
    if type_diffusion == 'exp':
        constraints.append(t_hat[c_act] == cp.sum(a_hat[idx_ord[:cidx[0]]]))
    elif type_diffusion == 'pl':
        tdifs = 1.0 / (val[cidx[0]] - val[:cidx[0]])
        indv = np.where(tdifs < 1)[0]
        tdifs = tdifs[indv]
        constraints.append(t_hat[c_act] <= cp.sum(
            tdifs * a_hat[idx_ord[indv]]))
    elif type_diffusion == 'rayleigh':
        tdifs = (val[cidx[0]] - val[:cidx[0]])
        constraints.append(t_hat[c_act] <= cp.sum(
            tdifs * a_hat[idx_ord[:cidx[0]]]))


def estimate_network(A, C, num_nodes, horizon, type_diffusion):
    num_cascades = np.zeros(num_nodes)
    A_potential = lil_matrix(np.zeros(A.shape))
    A_bad = lil_matrix(np.zeros(A.shape))
    A_hat = lil_matrix(np.zeros(A.shape))
    total_obj = 0

    for c in range(C.shape[0]):
        idx = np.where(C[c, :] != -1)[0]  # used nodes
        val = np.sort(C[c, idx])
        order = np.argsort(val)

        for i in range(1, len(val)):
            num_cascades[idx[order[i]]] += 1
            for j in range(i):
                diff = val[i] - val[j]
                add_log_survival(
                    A_potential, idx[order[j]], idx[order[i]], diff, type_diffusion)

        for j in range(num_nodes):
            if j not in idx:
                for i in range(len(val)):
                    diff = horizon - val[i]
                    add_log_survival(
                        A_bad, idx[order[i]], j, diff, type_diffusion)

    # convex program per column
    for i in range(num_nodes):
        if num_cascades[i] == 0:
            continue
        print(f'Processing node {i}...')
        a_hat = cp.Variable(num_nodes)
        t_hat = cp.Variable(int(num_cascades[i]))
        obj = 0

        potential_indices = A_potential[:, i].nonzero()[0]
        bad_indices = A_bad[:, i].nonzero()[0]

        constraints = [a_hat[potential_indices] >= 0]

        for j in potential_indices:
            obj += -a_hat[j] * (A_potential[j, i] + A_bad[j, i])

        c_act = 0
        for c in range(C.shape[0]):
            idx = np.where(C[c, :] != -1)[0]  # used nodes
            val = np.sort(C[c, idx])
            order = np.argsort(val)
            idx_ord = idx[order]
            cidx = np.where(idx_ord == i)[0]

            if cidx.size > 0 and cidx[0] > 0:
                add_constraints(constraints, a_hat, t_hat, c_act,
                                val, idx_ord, cidx, type_diffusion)
                obj += cp.log(t_hat[c_act])
                c_act += 1

        constraints += [a_hat >= 0]

        problem = cp.Problem(cp.Maximize(obj), constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        total_obj += problem.value
        A_hat[:, i] = a_hat.value

    return A_hat, total_obj
