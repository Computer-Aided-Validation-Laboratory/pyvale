#-------------------------------------------------------------------------
# pyvale: gmsh,monoblock,3mat,thermal/mechanical,transient
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

endTime = 20
timeStep = 1

coolantTemp= 150.0      # degC
heatTransCoeff= 125.0e3 # W.m^-2.K^-1
surfHeatFlux= 4.67e6    # W.m^-2, taken from Adel's first paper
timeConst = 1   # s

# Material Properties:
# Thermal Props: Copper-Chromium-Zirconium pg.148 at 200degC
cucrzrDensity = 8816.0  # kg.m^-3
cucrzrThermCond = 343.0 # W.m^-1.K^-1
cucrzrSpecHeat = 407.0  # J.kg^-1.K^-1

# Thermal Props: Pure (OFHC) Copper at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

# Thermal Props: Tungsten at 600degC
wDensity = 19150.0  # kg.m^-3
wThermCond = 127.0 # W.m^-1.K^-1
wSpecHeat = 147.0  # J.kg^-1.K^-1

# Mechanical Props: Copper-Chromium-Zirconium at 200degC
cucrzrEMod = 123e9       # Pa
cucrzrPRatio = 0.33      # -

# Mechanical Props: OFHC Copper at 250degC
cuEMod = 108e9       # Pa
cuPRatio = 0.33      # -

# Mechanical Props: Tungsten at 250degC
wEMod = 387e9       # Pa
wPRatio = 0.29      # -

# Thermo-mechanical coupling
stressFreeTemp = ${coolantTemp} # degC
cucrzrThermExp =  17.7e-6 # 1/degC
cuThermExp = 17.8e-6 # 1/degC
wThermExp = 4.72e-6 # 1/degC

# Mesh file string
mesh_file = 'case16.msh'

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[GlobalParams]
  displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
  type = FileMesh
  file = ${mesh_file}
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = SECOND
    initial_condition = ${coolantTemp}
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
  []
  [time_derivative]
    type = HeatConductionTimeDerivative
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
      generate_output = 'strain_xx strain_xy strain_xz strain_yx strain_yy strain_yz strain_zx strain_zy strain_zz'
  []
[]

[Materials]
    [cucrzr_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = ${cucrzrThermCond}
        specific_heat = ${cucrzrSpecHeat}
        block = 'pipe-cucrzr'
    []
    [cucrzr_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${cucrzrDensity}
        block = 'pipe-cucrzr'
    []
    [cucrzr_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${cucrzrEMod}
        poissons_ratio = ${cucrzrPRatio}
        block = 'pipe-cucrzr'
    []
    [cucrzr_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${cucrzrThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
        block = 'pipe-cucrzr'
    []


    [copper_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = ${cuThermCond}
        specific_heat = ${cuSpecHeat}
        block = 'interlayer-cu'
    []
    [copper_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${cuDensity}
        block = 'interlayer-cu'
    []
    [copper_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${cuEMod}
        poissons_ratio = ${cuPRatio}
        block = 'interlayer-cu'
    []
    [copper_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${cuThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
        block = 'interlayer-cu'
    []


    [tungsten_thermal]
        type = HeatConductionMaterial
        thermal_conductivity = ${wThermCond}
        specific_heat = ${wSpecHeat}
        block = 'armour-w'
    []
    [tungsten_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${wDensity}
        block = 'armour-w'
    []
    [tungsten_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${wEMod}
        poissons_ratio = ${wPRatio}
        block = 'armour-w'
    []
    [tungsten_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${wThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
        block = 'armour-w'
    []

    [stress]
        type = ComputeFiniteStrainElasticStress # ComputeLinearElasticStress or ComputeFiniteStrainElasticStress
    []
[]

[BCs]
  [heat_flux_in]
    type = FunctionNeumannBC
    variable = temperature
    boundary = 'bc-top-heatflux'
    function = '${fparse surfHeatFlux}*(1-exp(-(1/${timeConst})*t))'
    #function = '${fparse surfHeatFlux}*(t/${fparse endTime/2})'
  []
  [heat_flux_out]
    type = ConvectiveHeatFluxBC
    variable = temperature
    boundary = 'bc-pipe-heattransf'
    T_infinity = ${coolantTemp}
    heat_transfer_coefficient = ${heatTransCoeff}
  []

  # Lock disp_y for base
    # NOTE: if locking y on base need to comment all disp_y conditions below
    [mech_bc_c_dispy]
        type = DirichletBC
       variable = disp_y
        boundary = 'bc-base-disp'
        value = 0.0
    []

    # Lock all disp DOFs at the center of the block
    [mech_bc_c_dispx]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-c-point-xyz-mech'
        value = 0.0
    []
    #[mech_bc_c_dispy]
    #    type = DirichletBC
    #    variable = disp_y
    #    boundary = 'bc-c-point-xyz-mech'
    #    value = 0.0
    #[]
    [mech_bc_c_dispz]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-c-point-xyz-mech'
        value = 0.0
    []

    # Lock disp yz along the x (left-right) axis
    #[mech_bc_l_dispy]
    #    type = DirichletBC
    #    variable = disp_y
    #    boundary = 'bc-l-point-yz-mech'
    #    value = 0.0
    #[]
    [mech_bc_l_dispz]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-l-point-yz-mech'
        value = 0.0
    []
    #[mech_bc_r_dispy]
    #    type = DirichletBC
    #    variable = disp_y
    #    boundary = 'bc-r-point-yz-mech'
    #    value = 0.0
    #[]
    [mech_bc_r_dispz]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-r-point-yz-mech'
        value = 0.0
    []

    # Lock disp xy along the z (front-back) axis
    [mech_bc_f_dispx]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-f-point-xy-mech'
        value = 0.0
    []
    #[mech_bc_f_dispy]
    #    type = DirichletBC
    #    variable = disp_y
    #    boundary = 'bc-f-point-xy-mech'
    #    value = 0.0
    #[]
    [mech_bc_b_dispx]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-b-point-xy-mech'
        value = 0.0
    []
    #[mech_bc_b_dispy]
    #    type = DirichletBC
    #    variable = disp_y
    #    boundary = 'bc-b-point-xy-mech'
    #    value = 0.0
    #[]
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient

  solve_type = 'NEWTON'   # PJNFK or NEWTON
  petsc_options = '-snes_converged_reason'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
  line_search = 'none'

  l_max_its = 1000
  l_tol = 1e-6

  nl_max_its = 50
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6

  start_time=0.0
  end_time = ${endTime}
  dt = ${timeStep}

  [Predictor]
    type = SimplePredictor
    scale = 1
  []
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
  [disp_z_max]
    type = NodalExtremeValue
    variable = disp_z
  []

  [strain_xx_max]
      type = ElementExtremeValue
      variable = strain_xx
  []
  [strain_yy_max]
      type = ElementExtremeValue
      variable = strain_yy
  []
  [strain_zz_max]
    type = ElementExtremeValue
    variable = strain_zz
  []
[]


[Outputs]
  exodus = true
[]
