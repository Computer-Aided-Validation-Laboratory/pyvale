#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,2mat,thermal,steady,
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
coolantTemp = 20.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1
surfHeatFlux = 500.0e3    # W.m^-2

# Material Properties:
# Pure (OFHC) Copper at 250degC
cuDensity = 8829.0  # kg.m^-3
cuThermCond = 384.0 # W.m^-1.K^-1
cuSpecHeat = 406.0  # J.kg^-1.K^-1

# Tungsten at 600degC
wDensity = 19150.0  # kg.m^-3
wThermCond = 127.0 # W.m^-1.K^-1
wSpecHeat = 147.0  # J.kg^-1.K^-1

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Mesh]
  [generated]
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
    input = generated
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
[]

[Variables]
  [temperature]
    initial_condition = ${coolantTemp}
  []
[]

[Kernels]
  [heat_conduction]
    type = HeatConduction
    variable = temperature
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
[]

[BCs]
  [heat_flux_out]
      type = ConvectiveHeatFluxBC
      variable = temperature
      boundary = 'left'
      T_infinity = ${coolantTemp}
      heat_transfer_coefficient = ${heatTransCoeff}
  []
  [heat_flux_in]
      type = NeumannBC
      variable = temperature
      boundary = 'right'
      value = ${surfHeatFlux}
  []
[]

[Executioner]
  type = Steady
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
[]

[Outputs]
  exodus = true
[]