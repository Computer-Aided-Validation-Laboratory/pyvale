#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,2mat,thermomechanical,steady
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# Geometric Properties
lengX = 100e-3  # m
lengY = 50e-3   # m

# Mesh Properties
nElemX = 20
nElemY = 10
eType = QUAD4 # QUAD4 for 1st order, QUAD8 for 2nd order

# Thermal Loads/BCs
uniformTemp = 100 # degC

# Material Properties:
# Thermal Props: Pure (OFHC) Copper at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

# Thermal Props: Tungsten at 600degC
wDensity = 19150.0  # kg.m^-3
wThermCond = 127.0 # W.m^-1.K^-1
wSpecHeat = 147.0  # J.kg^-1.K^-1

# Mechanical Props: OFHC Copper at 250degC
cuEMod = 108e9       # Pa
cuPRatio = 0.33      # -

# Mechanical Props: Tungsten at 250degC
wEMod = 387e9       # Pa
wPRatio = 0.29      # -

# Thermo-mechanical coupling
stressFreeTemp = 20 # degC
cuThermExp = 17.8e-6 # 1/degC
wThermExp = 4.72e-6 # 1/degC

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

# Selection tolerance for bounding box node selection
sTol = ${fparse lengX/(nElemX*4)}

[GlobalParams]
  displacements = 'disp_x disp_y'
[]

[Mesh]
  [generated_mesh]
    type = GeneratedMeshGenerator
    dim = 2
    nx = ${nElemX}
    ny = ${nElemY}
    xmax = ${lengX}
    ymax = ${lengY}
    elem_type = ${eType}
  []

  [block1]
    type = SubdomainBoundingBoxGenerator
    input = generated_mesh
    block_id = 1
    bottom_left = '0 0 0'
    top_right = '${fparse lengX/2} ${fparse lengY} 0'
  []
  [block2]
    type = SubdomainBoundingBoxGenerator
    input = block1
    block_id = 2
    bottom_left = '${fparse lengX/2} 0 0'
    top_right = '${fparse lengX} ${fparse lengY} 0'
  []

  [node_bottom_left]
    type = BoundingBoxNodeSetGenerator
    input = block2
    bottom_left = '${fparse 0-sTol}
                   ${fparse 0-sTol}
                   ${fparse 0-sTol}'
    top_right = '${fparse 0+sTol}
                 ${fparse 0+sTol}
                 ${fparse 0+sTol}'
    new_boundary = bottom_left_node
  []
  [node_top_left]
      type = BoundingBoxNodeSetGenerator
      input = node_bottom_left
      bottom_left = '${fparse 0-sTol}
                    ${fparse lengY-sTol}
                    ${fparse 0-sTol}'
      top_right = '${fparse 0+sTol}
                  ${fparse lengY+sTol}
                  ${fparse 0+sTol}'
      new_boundary = top_left_node
  []
  [node_bottom_right]
      type = BoundingBoxNodeSetGenerator
      input = node_top_left
      bottom_left = '${fparse lengX-sTol}
                    ${fparse 0-sTol}
                    ${fparse 0-sTol}'
      top_right = '${fparse lengX+sTol}
                  ${fparse 0+sTol}
                  ${fparse 0+sTol}'
      new_boundary = bottom_right_node
  []

  [full_plate_nodeset]
    type = BoundingBoxNodeSetGenerator
    input = node_bottom_right
    bottom_left = '${fparse 0-sTol}
                  ${fparse 0-sTol}
                  ${fparse 0-sTol}'
    top_right = '${fparse lengX+sTol}
                ${fparse lengY+sTol}
                ${fparse 0+sTol}'
    new_boundary = full_plate
  []
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = FIRST
    initial_condition = ${uniformTemp}
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
[]

[Modules/TensorMechanics/Master]
  [all]
      strain = SMALL                      # SMALL or FINITE
      incremental = true
      add_variables = true
      material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
      material_output_order = FIRST       # CONSTANT, FIRST, SECOND,
      automatic_eigenstrain_names = true
      generate_output = 'vonmises_stress strain_xx strain_xy strain_xz strain_yx strain_yy strain_yz strain_zx strain_zy strain_zz stress_xx stress_xy stress_xz stress_yx stress_yy stress_yz stress_zx stress_zy stress_zz max_principal_strain mid_principal_strain min_principal_strain'
  []
[]

[Materials]
  [copper_thermal]
    type = HeatConductionMaterial
    thermal_conductivity = ${cuThermCond}
    specific_heat = ${cuSpecHeat}
    block = 1
  []
  [copper_density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = ${cuDensity}
    block = 1
  []
  [copper_elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${cuEMod}
    poissons_ratio = ${cuPRatio}
    block = 1
  []
  [copper_expansion]
    type = ComputeThermalExpansionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_coeff = ${cuThermExp}
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 1
  []

  [tungsten_thermal]
    type = HeatConductionMaterial
    thermal_conductivity = ${wThermCond}
    specific_heat = ${wSpecHeat}
    block = 2
  []
  [tungsten_density]
    type = GenericConstantMaterial
    prop_names = 'density'
    prop_values = ${wDensity}
    block = 2
  []
  [tungsten_elasticity]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${wEMod}
    poissons_ratio = ${wPRatio}
    block = 2
  []
  [tungsten_expansion]
    type = ComputeThermalExpansionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_coeff = ${wThermExp}
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 2
  []

  [stress]
    type = ComputeFiniteStrainElasticStress # ComputeLinearElasticStress or ComputeFiniteStrainElasticStress
  []
[]

[BCs]
  [uniform_temp]
    type = DirichletBC
    variable = temperature
    boundary = 'full_plate'
    value = ${uniformTemp}
  []

  [bottom_left_disp_x]
    type = DirichletBC
    variable = disp_x
    boundary = 'bottom_left_node'
    value = 0.0
  []
  [bottom_left_disp_y]
      type = DirichletBC
      variable = disp_y
      boundary = 'bottom_left_node'
      value = 0.0
  []
  [bottom_right_disp_y]
      type = DirichletBC
      variable = disp_y
      boundary = 'bottom_right_node'
      value = 0.0
  []
  [top_left_disp_x]
      type = DirichletBC
      variable = disp_x
      boundary = 'top_left_node'
      value = 0.0
  []
[]

[Preconditioning]
  [smp]
      type = SMP
      full = true
  []
[]

[Executioner]
  type = Steady
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre    boomeramg'
  #end_time= ${endTime}
  #dt = ${timeStep}
[]


[Postprocessors]
  [temp_max]
      type = NodalExtremeValue
      variable = temperature
  []
  [temp_avg]
      type = AverageNodalVariableValue
      variable = temperature
  []

  [disp_x_max]
      type = NodalExtremeValue
      variable = disp_x
  []
  [disp_y_max]
      type = NodalExtremeValue
      variable = disp_y
  []

  [strain_xx_max]
      type = ElementExtremeValue
      variable = strain_xx
  []
  [strain_yy_max]
      type = ElementExtremeValue
      variable = strain_yy
  []

  [strain_xx_avg]
      type = ElementAverageValue
      variable = strain_xx
  []
  [strain_yy_avg]
      type = ElementAverageValue
      variable = strain_yy
  []
[]

[Outputs]
  exodus = true
[]