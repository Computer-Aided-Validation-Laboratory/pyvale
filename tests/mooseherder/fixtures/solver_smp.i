# Rabbit serial/SMP builds do not provide Hypre.
[Executioner]
  solve_type = NEWTON
  automatic_scaling = false
  petsc_options_iname = '-pc_type -ksp_type -ksp_gmres_restart'
  petsc_options_value = 'lu gmres 30'
  l_max_its = 100
  l_tol = 1e-6
  nl_max_its = 50
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
[]
