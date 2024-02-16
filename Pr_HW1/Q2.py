import cvxpy as cp
from disks_data import *
import numpy as np


def main():

    C = cp.Variable((n, 2))
    R = cp.Variable(n)

    constraints = [R >= 0]
    constraints += [C[:k, :] == Cgiven[:k, :]]
    constraints += [R[:k] == Rgiven[:k]]

    for i in range(len(Gindexes)):
        constraints += [cp.norm(C[Gindexes[i, 0], :] - C[Gindexes[i, 1], :])
                        <= R[Gindexes[i, 0]] + R[Gindexes[i, 1]]]

    area_objective = cp.Minimize(np.pi * cp.sum_squares(R))
    perimeter_objective = cp.Minimize(2 * np.pi * cp.sum(R))

    min_area_problem = cp.Problem(area_objective, constraints)
    min_perimeter_problem = cp.Problem(perimeter_objective, constraints)

    min_area_problem.solve()
    print('Optimal area: ', min_area_problem.value)
    plot_disks(C.value, R.value, Gindexes, name="areas.png")

    min_perimeter_problem.solve()
    print('Optimal perimeter: ', min_perimeter_problem.value)
    plot_disks(C. value, R.value, Gindexes, name="perimeters.png")


if __name__ == '__main__':
    main()
