# Serial/SMP defaults follow Rabbit's thermal and monoblock solvers.
# Time integration and physical data remain defined in each case.
[Executioner]
  solve_type = NEWTON
  petsc_options_iname = '-pc_type -ksp_type -ksp_gmres_restart'
  petsc_options_value = 'ilu gmres 30'
  automatic_scaling = true
  l_max_its = 100
  l_tol = 1e-6
  nl_max_its = 50
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
[]
