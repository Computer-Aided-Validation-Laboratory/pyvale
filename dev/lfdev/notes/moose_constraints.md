[Constraints]
  [./y_top]
    type = EqualValueBoundaryConstraint
    variable = disp_y
    secondary = 'Top-BC' # boundary
    penalty = 10e3
  [../]
[]