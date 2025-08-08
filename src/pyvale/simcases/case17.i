#-------------------------------------------------------------------------
# pyvale: simple,2DplateWHole,mechanical,transient
#-------------------------------------------------------------------------
# NOTE: default 2D MOOSE solid mechanics is plane strain

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

endTime = 60
timeStep = 1

# Mechanical Loads/BCs
topDispRate = ${fparse 1.5e-3 / 60}  # m/s

# Material Properties: OFHC Copper 250degC
EMod= 108e9   # Pa
PRatio = 0.33     # -

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    type = FileMesh
    file = 'case17.msh'
[]

[Modules/TensorMechanics/Master]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
        material_output_order = FIRST       # CONSTANT, FIRST, SECOND,
        generate_output = 'vonmises_stress stress_xx stress_yy stress_xy strain_xx strain_yy strain_xy'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-base'
        value = 0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bc-base'
        value = 0
    []


    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-top'
        value = 0.0
    []
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'bc-top'
        function = '${topDispRate}*t'
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${EMod}
        poissons_ratio = ${PRatio}
    []
    [stress]
        type = ComputeFiniteStrainElasticStress
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
        boundary = 'bc-base'
    []
    [react_y_top]
        type = SidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bc-top'
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