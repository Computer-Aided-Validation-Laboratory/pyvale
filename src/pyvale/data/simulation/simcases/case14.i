#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,mechanical,steady,
#-------------------------------------------------------------------------
# NOTE: default 2D MOOSE solid mechanics is plane strain

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

# NOTE: only used for transient solves
endTime = 60    # s
timeStep = 5    # s

# Geometric Properties
lengX = 50e-3  # m
lengY = 100e-3   # m

# Mesh Properties
nElemX = 10
nElemY = 20
eType = QUAD8 # QUAD4 for 1st order, QUAD8 for 2nd order

# Mechanical Loads/BCs
topDispRate = ${fparse 1e-3 / 60}  # m/s

# Material Properties: OFHC Copper 250degC
cuEMod= 108e9   # Pa
cuPRatio = 0.33     # -

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

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
[]

[Modules/TensorMechanics/Master]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
        material_output_order = FIRST       # CONSTANT, FIRST, SECOND,
        generate_output = 'vonmises_stress stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy max_principal_strain mid_principal_strain min_principal_strain'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bottom'
        value = 0.0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bottom'
        value = 0.0
    []
    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'top'
        value = 0.0
    []
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'top'
        function = '${topDispRate}*t'
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${cuEMod}
        poissons_ratio = ${cuPRatio}
    []
    [stress]
        type = ComputeFiniteStrainElasticStress # ComputeLinearElasticStress
    []
[]

[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient
    solve_type = 'PJFNK'
    petsc_options_iname = '-pc_type -pc_hypre_type'
    petsc_options_value = 'hypre boomeramg'
    end_time= ${endTime}
    dt = ${timeStep}
[]


[Postprocessors]
    [react_y_bot]
        type = SidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bottom'
    []
    [react_y_top]
        type = SidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'top'
    []

    [disp_y_max]
        type = NodalExtremeValue
        variable = disp_y
    []
    [disp_x_max]
        type = NodalExtremeValue
        variable = disp_x
    []

    [max_yy_stress]
        type = ElementExtremeValue
        variable = stress_yy
    []

    [strain_yy_avg]
        type = ElementAverageValue
        variable = strain_yy
    []
    [strain_xx_avg]
        type = ElementAverageValue
        variable = strain_xx
    []

    [stress_vm_max]
        type = ElementExtremeValue
        variable = vonmises_stress
    []
[]

[Outputs]
    exodus = true
[]