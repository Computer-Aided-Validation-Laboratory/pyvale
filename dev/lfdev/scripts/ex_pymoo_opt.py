# EXAMPLE: pymoo optimisation
import numpy as np
import matplotlib.pyplot as plt

from pymoo.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize

from pyvale.visualopts import PlotOptsGeneral
import dev.lfdev.optimcheckfuncs as cf

#-------------------------------------------------------------------------------
def cost_func(x):
    return cf.rastrigin(x)

class TestProblem(Problem):
    def __init__(self, n_var=2, xl=-10.0, xu=10.0):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=float)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = cost_func(x)

#-------------------------------------------------------------------------------
def main() -> None:
    xl = -10.0
    xu = 10.0
    problem = TestProblem(xl=xl,xu=xu)

    alg = 'GA'
    if alg == 'PSO':
        algorithm = PSO()
    else:
        algorithm = GA(
            pop_size=100,
            eliminate_duplicates=True)

    termination = DefaultSingleObjectiveTermination(
        xtol=1e-8,
        cvtol=1e-6,
        ftol=1e-6,
        period=10, # Number of generations that need to be below tols
        n_max_gen=100,
        n_max_evals=100000)

    res = minimize(problem,
                algorithm,
                termination,
                #seed=1,
                verbose=True)

    print(80*'=')
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    print(80*'=')

    pp = PlotOptsGeneral()
    (fig,ax) = cf.plot_fun_2d(f'Min with {alg}',
                              cost_func,(xl,xu),(xl,xu),100)
    plt.plot(res.X[0],res.X[1],'+r',lw=pp.lw,ms=pp.ms)
    plt.show()

if __name__ == "__main__":
    main()