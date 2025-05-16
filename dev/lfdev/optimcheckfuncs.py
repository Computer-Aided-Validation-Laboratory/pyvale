"""
OPTIMISATION TEST FUNCTIONS - N DIMS
https://gist.github.com/denis-bz/da697d8bc74fae4598bf
https://www.sfu.ca/~ssurjano/optimization.html
https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
from typing import Callable,Any
import numpy as np
import matplotlib.pyplot as plt
from pyvale.visualopts import PlotOptsGeneral


def ackley(x: np.ndarray,
             a: float = 20.0,
             b: float = 0.2,
             c: float = 2*np.pi) -> np.ndarray:
    """ACKLEY
    Dimension: N
    Local Minima: many
    Global Minimum: f(x) = 0 @ (0,0,....,0)
    Eval: [-32.768,32.768] or smaller
    """
    n = x.shape[1]
    sum1 = np.sum(x**2,axis=1)
    sum2 = np.sum(np.cos(c*x),axis=1)
    f =  -a*np.exp(-b*np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.exp(1)
    return f


def dixonprice(x: np.ndarray) -> np.ndarray:
    """DIXON-PRICE
    Dimension: N
    Local Minima: Large valley
    Global Minimum: f(x) = 0 @ x_i = 2^-((2^i-2)/2^i) for i = 1,...,d
    Eval: [-10.0,10.0] or smaller
    """
    n = x.shape[1]
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    f = np.sum( j * (x2[:,1:] - x[:,:-1]) **2, axis=1) + (x[:,0] - 1) **2
    return f


def griewank(x: np.ndarray, div: float = 4000.0) -> np.ndarray:
    """GRIEWANK
    Dimension: N
    Local Minima: many
    Global Minimum: f(x) = 0 @ (0,0,....,0)
    Eval: [-600,600] or smaller
    """
    n = x.shape[1]
    j = np.arange( 1., n+1 )
    sum1 = np.sum( x**2, axis=1 )
    p = np.prod( np.cos( x / np.sqrt(j) ), axis=1)
    f = sum1/div - p + 1
    return f


def rastrigin(x: np.ndarray, a: float = 10) -> np.ndarray:
    """RASTRIGIN
    Dimension: N
    Local Minima: many
    Global Minimum: f(x) = 0 @ (0,0,....,0)
    Eval: [-5.12,5.12] or smaller
    """
    n = x.shape[1]
    sum1 = np.sum(x**2 - a*np.cos(2*np.pi*x),axis=1)
    f = a*n + sum1
    return f


def rosenbrock(x: np.ndarray, a: float = 100) -> float:
    """ROSENBROCK
    Dimension: N
    Local Minima: Large valley
    Global minimum: at (x,y) = (1,1) where f(x,y)=0
    Eval: [-5.0,10.0] or smaller
    """
    x0 = x[:,:-1] # x_(i) ... to n-1
    x1 = x[:,1:]  # x_(i+1) ... to n-1
    f = a*np.sum((x1-x0**2)**2,axis=1) + np.sum((1-x0)**2, axis=1)
    return f


def sphere(x: np.ndarray) -> np.ndarray:
    """SPHERE
    Dimension: N
    Local Minima: none
    Global minimum: f(x) = 0 @ (0,0,....,0)
    Eval: [-inf,inf] or smaller
    """
    f = np.sum(x**2,axis=1)
    return f


def get_mesh_x_2d(xlim1: tuple[float,float],
                  xlim2: tuple[float,float],
                  n: int = 100):
    xv1 = np.linspace(xlim1[0],xlim1[1],n)
    xv2 = np.linspace(xlim2[0],xlim2[1],n)
    (xm1,xm2) = np.meshgrid(xv1,xv2)
    return (xm1,xm2)


def get_flat_x_2d(xlim1: tuple[float,float],
                  xlim2: tuple[float,float],
                  n: int = 100) -> np.ndarray:

    (xm1,xm2) = get_mesh_x_2d(xlim1,xlim2,n)
    xf1 = xm1.flatten()
    xf2 = xm2.flatten()
    return np.column_stack((xf1,xf2))


def f_mesh_2d(fun: Callable,
              xlim1: tuple[float,float],
              xlim2: tuple[float,float],
              n: int = 100) -> np.ndarray:

    (xm1,_) = get_mesh_x_2d(xlim1,xlim2,n)
    xf = get_flat_x_2d(xlim1,xlim2,n)
    f_flat = fun(xf)
    f_mesh = f_flat.reshape(xm1.shape)

    return f_mesh


def plot_fun_2d(tStr: str,
                fun: Callable,
                xlim1: tuple[float,float],
                xlim2: tuple[float,float],
                n: int =100) -> tuple[Any,Any]:

    (xm1,xm2) = get_mesh_x_2d(xlim1,xlim2,n)
    f_mesh = f_mesh_2d(fun,xlim1,xlim2,n)
    # Plot the function
    pp = PlotOptsGeneral()
    fig, ax = plt.subplots(figsize=pp.single_fig_size_square,
                           layout="constrained")
    fig.set_dpi(pp.resolution)

    plt.contourf(xm1, xm2, f_mesh, 20, cmap=pp.cmap_seq)

    plt.colorbar()
    plt.title(tStr,fontsize=pp.font_head_size,fontname=pp.font_name)
    plt.xlabel("x1",fontsize=pp.font_ax_size,fontname=pp.font_name)
    plt.ylabel("x2",fontsize=pp.font_ax_size,fontname=pp.font_name)

    #plt.show()
    #plt.savefig(save_path+save_name, dpi=pp.resolution, format="png", bbox_inches="tight")
    return fig,ax


