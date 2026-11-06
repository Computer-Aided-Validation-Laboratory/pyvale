# Replace serial preconditioning; all physical and time parameters are shared.
[Executioner]
  petsc_options_iname := '-pc_type -pc_hypre_type -ksp_type -ksp_gmres_restart'
  petsc_options_value := 'hypre boomeramg gmres 30'
[]
