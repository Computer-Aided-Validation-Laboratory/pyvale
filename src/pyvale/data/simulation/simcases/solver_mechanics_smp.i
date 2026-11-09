# Rabbit's small elastic plate cases use LU in SMP and MPI configurations.
!include solver_smp.i
[Executioner]
  automatic_scaling := false
  petsc_options_value := 'lu gmres 30'
[]
