import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

T = 24
N = np.array([0, 4, 2, 2, 3, 0, 4, 5, 6, 6, 4, 1,
              4, 4, 0, 1, 3, 4, 2, 0, 3, 2, 0, 1])
Ntest = np.array([0, 1, 3, 2, 3, 1, 4, 5, 3, 1, 4, 3,
                  5, 5, 2, 1, 1, 1, 2, 0, 1, 2, 1, 0])


def main():
    _lambda = cp.Variable(T)
    rho = cp.Parameter(nonneg=True)

    constraints = [_lambda >= 0]

    objective = cp.Minimize(cp.sum(_lambda) - N @ cp.log(_lambda) + rho * (
        cp.sum_squares(cp.diff(_lambda)) + (_lambda[0] - _lambda[-1])**2))

    prob = cp.Problem(objective, constraints)

    rho_ls = [0.1, 1, 10, 100]
    _lambda_ls = []
    for r in rho_ls:
        rho.value = r
        prob.solve()
        _lambda_ls.append(_lambda.value)
        plt.plot(np.arange(T), _lambda.value, label="rho=%.1f" % r)
    plt.legend()
    plt.savefig("preodic_poisson.png", dpi=300)

    for lam, r in zip(_lambda_ls, rho_ls):
        logprob = np.sum(
            np.log(np.exp(-lam) * lam ** Ntest / factorial(Ntest)))
        print(f"Rho = {r}, log-likelihood = {logprob}")


if __name__ == '__main__':
    main()
