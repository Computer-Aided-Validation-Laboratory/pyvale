#-------------------------------------------------------------------------
# pyvale: gmsh,3Dstcgmsh,1mat,thermomechanical,steady,
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# Thermal Loads/BCs
coolantTemp = 100.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1
surfHeatFlux = 5.0e6    # W.m^-2

# Material Properties:
# Thermal Props:OFHC) Copper at 250degC
ss316LDensity = 8829.0  # kg.m^-3
ss316LThermCond = 384.0 # W.m^-1.K^-1
ss316LSpecHeat = 406.0  # J.kg^-1.K^-1

# Mechanical Props: OFHC Copper 250degC
ss316LEMod = 108e9       # Pa
ss316LPRatio = 0.33      # -

# Thermo-mechanical coupling
stressFreeTemp = 150 # degC
cuThermExp = 17.8e-6 # 1/degC

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------


[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'case10.msh'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = FIRST
        initial_condition = ${coolantTemp}
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
        thermal_conductivity = ${ss316LThermCond}
        specific_heat = ${ss316LSpecHeat}
    []
    [copper_density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = ${ss316LDensity}
    []
    [copper_elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${ss316LEMod}
        poissons_ratio = ${ss316LPRatio}
    []
    [copper_expansion]
        type = ComputeThermalExpansionEigenstrain
        temperature = temperature
        stress_free_temperature = ${stressFreeTemp}
        thermal_expansion_coeff = ${cuThermExp}
        eigenstrain_name = thermal_expansion_eigenstrain
    []

    [stress]
        type = ComputeFiniteStrainElasticStress # ComputeLinearElasticStress or ComputeFiniteStrainElasticStress
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

    #[strain_xx_avg]
    #    type = ElementAverageValue
    #    variable = strain_xx
    #[]
    #[strain_yy_avg]
    #    type = ElementAverageValue
    #    variable = strain_yy
    #[]
[]

[Outputs]
    exodus = true
[]