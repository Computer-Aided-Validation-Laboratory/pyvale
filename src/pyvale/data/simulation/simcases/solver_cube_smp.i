# Rabbit's small thermo-mechanical cube uses tighter nonlinear tolerances.
!include solver_smp.i
[Executioner]
  nl_max_its := 20
  nl_rel_tol := 1e-7
  nl_abs_tol := 1e-7
[]
