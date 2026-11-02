#-------------------------------------------------------------------------
# pyvale: gmsh,3Dstc,1mat,thermal,steady,
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# NOTE: only used for transient solves
endTime= 1
timeStep = 1

# Thermal Loads/BCs
coolantTemp = 150.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1

surfHeatPower = 0.25e3     # W
blockLeng = 49.5e-3
blockWidth = 37e-3
surfArea = ${fparse blockLeng*blockWidth}   # m^2
surfHeatFlux = ${fparse surfHeatPower/surfArea} # W.m^-2

# Material Properties: SS316L @ 400 degC
ss316LDensity = 7770.0  # kg.m^-3
ss316LThermCond = 19.99 # W.m^-1.K^-1
ss316LSpecHeat = 556.0  # J.kg^-1.K^-1

# Mesh file string
mesh_file = 'case19.msh'

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[Mesh]
    type = FileMesh
    file = ${mesh_file}
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
        thermal_conductivity = ${ss316LThermCond}
        specific_heat = ${ss316LSpecHeat}
    []
    [copper_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${ss316LDensity}
    []
[]

[BCs]
    [heat_flux_out]
        type = ConvectiveHeatFluxBC
        variable = temperature
        boundary = 'bc-pipe-htc'
        T_infinity = ${coolantTemp}
        heat_transfer_coefficient = ${heatTransCoeff}
    []
    [heat_flux_in]
        type = NeumannBC
        variable = temperature
        boundary = 'bc-top-heatflux'
        value = ${surfHeatFlux}
    []
[]

[Executioner]
    type = Transient
    end_time= ${endTime}
    dt = ${timeStep}
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