[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  type = FileMesh
  file = 'plate_2d_rectangle.msh'
[]

[Modules/TensorMechanics/Master]
  [all]
    add_variables = true
  []
[]

[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = 1e9
    poissons_ratio = 0.3
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]

[Executioner]
  type = Transient
  end_time = 2
  dt = 1
[]

[Outputs]
  exodus = true
[]




