#-------------------------------------------------------------------------
# pyvale: simple,2Dplate,mechanical,steady,
#-------------------------------------------------------------------------
# NOTE: default 2D MOOSE solid mechanics is plane strain

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START
endTime = 10
timeStep = 1

# Geometric Properties
lengX = 100e-3  # m
lengY = 150e-3   # m

# Mesh Properties
nElemX = ${fparse 10*2}
nElemY = ${fparse 15*2}
eType = QUAD8 # QUAD4 for 1st order, QUAD8 for 2nd order

# Mechanical Loads/BCs
dispRate = ${fparse 1.5e-3 / endTime}

# Material Properties: steel
steelEMod= 200e9   # Pa
steelPRatio = 0.3     # -

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
        generate_output = 'strain_xx strain_yy strain_xy'
    []
[]

[BCs]
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'top'
        function= '${dispRate}*t'
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bottom'
        value = 0.0
    []
    [left_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'left'
        value = 0.0
    []
    [right_x]
        type = FunctionDirichletBC
        variable = disp_x
        boundary = 'right'
        function = '${dispRate}*t'
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${steelEMod}
        poissons_ratio = ${steelPRatio}
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

[]

[Outputs]
    exodus = true
[]