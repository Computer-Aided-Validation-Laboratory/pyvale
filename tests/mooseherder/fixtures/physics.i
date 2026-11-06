[GlobalParams]
  displacements = 'disp_x disp_y'
[]
[Physics/SolidMechanics/QuasiStatic/all]
  strain = SMALL
  add_variables = true
  generate_output = 'stress_xx stress_yy strain_xx strain_yy'
  material_output_family = MONOMIAL
  material_output_order = CONSTANT
[]
[BCs]
  [left_x]
    type = DirichletBC
    variable = disp_x
    boundary = left
    value = 0
  []
  [bottom_y]
    type = DirichletBC
    variable = disp_y
    boundary = bottom
    value = 0
  []
  [top_y]
    type = FunctionDirichletBC
    variable = disp_y
    boundary = top
    function = '${topDisp}*t/${endTime}'
  []
[]
!include materials/elastic.i
[Preconditioning/SMP]
  type = SMP
  full = true
[]
[Executioner]
  type = Transient
  end_time = ${endTime}
  dt = ${timeStep}
[]
[Outputs]
  exodus = true
[]
