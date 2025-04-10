'''
================================================================================
Analytic test case data - linear

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import scipy.integrate
import mooseherder as mh
import pyvale
import pyvale.core.visualisation.visualplotters


def rectangle_mesh_2d(leng_x: float,
                      leng_y: float,
                      n_elem_x: int,
                      n_elem_y: int) -> tuple[np.ndarray,np.ndarray]:

    n_elems = n_elem_x*n_elem_y
    n_node_x = n_elem_x+1
    n_node_y = n_elem_y+1
    nodes_per_elem = 4

    coord_x = np.linspace(0,leng_x,n_node_x)
    coord_y = np.linspace(0,leng_y,n_node_y)
    (coord_grid_x,coord_grid_y) = np.meshgrid(coord_x,coord_y)

    coord_x = np.atleast_2d(coord_grid_x.flatten()).T
    coord_y = np.atleast_2d(coord_grid_y.flatten()).T
    coord_z = np.zeros_like(coord_x)
    coords = np.hstack((coord_x,coord_y,coord_z))

    connect = np.zeros((n_elems,nodes_per_elem)).astype(np.int64)
    row = 1
    nn = 0
    for ee in range(n_elems):
        nn += 1
        if nn >= row*n_node_x:
            row += 1
            nn += 1

        connect[ee,:] = np.array([nn,nn+1,nn+n_node_x+1,nn+n_node_x])
    connect = connect.T

    return (coords,connect)


def scalar_linear2d(coord_y: np.ndarray | float,
                coord_x: np.ndarray | float,
                time: np.ndarray | float = 1.0,
                x_grad: float = 10.0,
                y_grad: float = 10.0,
                offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((x_grad*coord_x +
               y_grad*coord_y))*time + offset
    return f_eval


def scalar_linear2d_int(coord_y: np.ndarray | float,
                    coord_x: np.ndarray | float,
                    time: np.ndarray | float = 1.0,
                    x_grad: float = 10.0,
                    y_grad: float = 10.0,
                    offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((x_grad/2 * coord_x**2 * coord_y +
              y_grad/2 * coord_y**2 * coord_x)*time +
              offset*coord_x*coord_y)
    return f_eval


def scalar_quad2d(coord_y: np.ndarray | float,
                coord_x: np.ndarray | float,
                time: np.ndarray | float = 1.0,
                roots_x: tuple[float,float] = (0.0,10.0),
                roots_y: tuple[float,float] = (0.0,10.0),
                offset: float = 20.0) -> np.ndarray | float:

    f_eval = ((coord_x - roots_x[0]) * (coord_x - roots_x[1]) *
              (coord_y - roots_y[0]) * (coord_y - roots_y[1]))*time + offset
    return f_eval

def scalar_fun(f_x,f_y,f_t):
    return 0


def main() -> None:
    #===========================================================================
    # User defined test case parameters
    leng_x = 10
    leng_y = 7.5
    n_elem_x = 4*10
    n_elem_y = 3*10
    time_steps = np.linspace(0.0,1.0,11)
    field_key = 'temperature'
    x_grad_coeff = 10.0
    y_grad_coeff = 10.0
    field_offset = 20.0

    roots_x = (0.0,leng_x)
    roots_y = (0.0,leng_y)
    #===========================================================================

    (xx,yy) = (0,1)
    (coords,connect) = rectangle_mesh_2d(leng_x,leng_y,n_elem_x,n_elem_y)


    # shape=(n_nodes,n_timesteps)
    '''
    scalar_vals = scalar_linear2d(np.atleast_2d(coords[:,yy]).T,
                                  np.atleast_2d(coords[:,xx]).T,
                                  np.atleast_2d(time_steps),
                                  x_grad = x_grad_coeff/leng_x,
                                  y_grad = y_grad_coeff/leng_y,
                                  offset = field_offset)

    '''
    scalar_vals = scalar_quad2d(np.atleast_2d(coords[:,yy]).T,
                                np.atleast_2d(coords[:,xx]).T,
                                np.atleast_2d(time_steps),
                                roots_x = roots_x,
                                roots_y = roots_y,
                                offset = field_offset)
    scalar_vals = np.array(scalar_vals)

    sim_data = mh.SimData()
    sim_data.num_spat_dims = 2
    sim_data.coords = coords
    sim_data.connect = {'connect1': connect}
    sim_data.node_vars = {field_key: scalar_vals}

    # Visualisation
    grid_shape = (n_elem_y+1,n_elem_x+1)
    coord_grid_x = np.atleast_2d(coords[:,xx]).T.reshape(grid_shape)
    coord_grid_y = np.atleast_2d(coords[:,yy]).T.reshape(grid_shape)
    scalar_grid = np.reshape(scalar_vals[:,-1],grid_shape)
    fig, ax = plt.subplots()
    cs = ax.contourf(coord_grid_x,
                     coord_grid_y,
                     scalar_grid)
    cbar = fig.colorbar(cs)
    plt.axis('scaled')
    #plt.show()

    pv_plot = pyvale.core.visualisation.visualplotters.plot_sim_data(sim_data,
                                                          field_key,
                                                          time_step=-1)
    #pv_plot.show(cpos="xy")

    # Integration
    x = sym.Symbol('x')
    y = sym.Symbol('y')
    sympy_f = ((x-roots_x[0])*(x-roots_x[1])*(y-roots_y[0])*(y-roots_y[1]) + # type: ignore
               field_offset)
    sympy_int = sym.integrate(sym.integrate(sympy_f,y), x)
    sympy_int_eval = (sympy_int.subs([(x,leng_x),(y,leng_y)]).evalf() - # type: ignore
                      sympy_int.subs([(x,0.0),(y,0.0)]).evalf())        # type: ignore

    print(f'Sympy integral = {sympy_int_eval}')

    scipy_int = scipy.integrate.dblquad(scalar_quad2d,0,leng_x,0,leng_y,
                                        args=(1.0,
                                              roots_x,
                                              roots_y,
                                              field_offset))
    print(f'Scipy integral = {scipy_int[0]}')

    print()
    print(f'{type(x)=}')
    print(f'{type(sympy_f)=}')
    print(f'{type(sympy_int)=}')
    print(f'{type(sympy_int_eval)=}')
    print(isinstance(sympy_f,sym.Expr))
    print()

if __name__ == '__main__':
    main()
