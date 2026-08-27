#-------------------------------------------------------------------------
# pyvale: gmsh,3Dstcgmsh,1mat,thermomechanical,steady,
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# Thermal Loads/BCs
coolantTemp = 150.0      # degC
heatTransCoeff = 125.0e3 # W.m^-2.K^-1

surfHeatPower = 0.25e3     # W
blockLeng = 49.5e-3
blockWidth = 37e-3
surfArea = ${fparse blockLeng*blockWidth}   # m^2
surfHeatFlux = ${fparse surfHeatPower/surfArea} # W.m^-2

# Material Properties:
# Thermal Props: SS316L @ 400degC
ss316LDensity = 7770.0  # kg.m^-3
ss316LThermCond = 19.99 # W.m^-1.K^-1
ss316LSpecHeat = 556.0  # J.kg^-1.K^-1

# Mechanical Props: SS316L @ 400degC
ss316LEMod = 168e9       # Pa
ss316LPRatio = 0.3      # -

# Thermo-mechanical coupling
stressFreeTemp = 150 # degC
ss316LThermExp = 17.8e-6 # 1/degC

# Element properties
elem_order = 'SECOND'

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------


[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'case20.msh'
[]

[Variables]
    [temperature]
        family = LAGRANGE
        order = ${elem_order}
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
        thermal_expansion_coeff = ${ss316LThermExp}
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

    # Lock disp_y for whole base
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
        boundary = 'bc-base-c-loc-xyz'
        value = 0.0
    []
    [mech_bc_c_dispz]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-base-c-loc-xyz'
        value = 0.0
    []

    # Lock z dof along x axis
    [mech_bc_px_dispz]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-base-nx-loc-z'
        value = 0.0
    []

    # Lock x dof along z
    [mech_bc_pz_dispx]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-base-pz-loc-x'
        value = 0.0
    []
    [mech_bc_nz_dispx]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-base-nz-loc-x'
        value = 0.0
    []
[]

[Preconditioning]
   [smp]
       type = SMP
       full = true
   []
[]

# LF-PersonalLaptop AMD 8 core / 8 thread
# Trans, Precon=ON, NEWTON, pctype=lu,  solve time with 7 mpi tasks = 229.18s
# Trans, Precon=OFF, NEWTON, pctype=lu,  solve time with 7 mpi tasks = 226.52s
# Steady, Precon=OFF, NEWTON, pctype=lu,  solve time with 7 mpi tasks = 226.52s

# LF-WorkLaptop AMD 8 core/ 16 threads
# Steady, Precon=OFF, NEWTON, pctype=lu,  solve time with 8 mpi tasks = 275s
# Steady, Precon=OFF, NEWTON, pctype=lu,  solve time with 6 mpi tasks = ~350s
# Steady, Precon=OFF, NEWTON, pctype=lu,  solve time with 4 mpi tasks = 458s

[Executioner]
    type = Steady
    #---------------------------------------------------------------------------
    solve_type = PJFNK   # PJNFK or NEWTON
    l_max_its = 100       # default = 1000
    nl_max_its = 100
    l_tol = 1e-5          # default = 1e-5
    nl_abs_tol = 1e-6     # default = 1e-50, set 1e-6
    nl_rel_tol = 1e-6     # default = 1e-8, set 1e-6

    line_search = none
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'

    #---------------------------------------------------------------------------
    # solve_type = 'NEWTON' # NEWTON or PJNFK
    # petsc_options_iname = '-pc_type'
    # petsc_options_value = 'lu'

    # l_max_its = 100
    # nl_max_its = 100
    # nl_rel_tol = 1e-6
    # nl_abs_tol = 1e-6
    # l_tol = 1e-5

    #---------------------------------------------------------------------------
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
[]

[Outputs]
    exodus = true
[]