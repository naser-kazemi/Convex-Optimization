import cvxpy as cp
from cvxpy import power, inv_pos, geo_mean


def main():
    x = cp.Variable()
    y = cp.Variable()

    def g_1(u, v): return geo_mean(cp.vstack([x, y]))
    def g_2(u): return inv_pos(u)
    def g_3(u): return power(u, 2)

    objective = cp.Minimize(g_3(g_2(g_1(x, y))))
    constraints = [0 <= x, 0 <= y]

    prob = cp.Problem(objective, constraints)

    prob.solve(cp.ECOS)

    print("status: ", prob.status)
    print("optimal value: ", prob.value)
    print("optimal var: ", x.value, y.value)


if __name__ == "__main__":
    main()
