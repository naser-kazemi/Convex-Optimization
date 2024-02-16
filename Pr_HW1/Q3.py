import cvxpy as cp
from zero_crossings_data import *
import matplotlib.pyplot as plt


def main():
    x = cp.Variable(2 * B)

    t_vector = np.arange(1, n + 1) / n
    frequency_vector = f_min + np.arange(B)

    C = np.cos(2 * np.pi * frequency_vector * t_vector[:, np.newaxis])
    S = np.sin(2 * np.pi * frequency_vector * t_vector[:, np.newaxis])

    A = np.hstack((C, S))

    objective = cp.Minimize(cp.norm(A @ x))

    constraints = [cp.multiply(s, A @ x) >= 0, s.T @ (A @ x) == n]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    y_hat = A @ x.value
    recovery_error = np.linalg.norm(y - y_hat) / np.linalg.norm(y)
    print(f"Recovery Error: {recovery_error}")

    plt.figure()
    plt.plot(np.arange(0, n), y, label="y")
    plt.plot(np.arange(0, n), y_hat, label="y_hat")
    plt.xlim([0, n])
    plt.legend(loc='lower left')
    plt.savefig("recovery_error.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
