
- initialise optimiser
	- define initial values, bounds, optimisation settings etc
	- run optimisation (pass in init values and get out dofs that minimise the objectiveFunction)

# pysci 
scipy.optimize.
least_squares

method{‘trf’, ‘dogbox’, ‘lm’}, optional

    Algorithm to perform minimization.

        ‘trf’ : Trust Region Reflective algorithm, particularly suitable for large sparse problems with bounds. Generally robust method.

        ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical use case is small problems with bounds. Not recommended for problems with rank-deficient Jacobian.

        ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle bounds and sparse Jacobians. Usually the most efficient method for small unconstrained problems.





def computeObjectiveFunction (in: dofs   (and info on strain, param scheme   out: scalar value assessing how good stress field is)
- reconstruct param maps using dofs and param scheme
- compute stress (function using strain and K)
- evaluate metrics
	- global_virtual_fields_cost_function.py   (sbvf or manual vfs)
		- TODO egi_cost_function.py
	- TODO fre_cost_function.py
- aggregate to single scalar  (dot product)
