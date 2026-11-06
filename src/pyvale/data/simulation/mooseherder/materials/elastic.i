[Materials]
  [elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${EMod}
    poissons_ratio = ${PRatio}
  []
  [stress]
    type = ComputeLinearElasticStress
  []
[]
