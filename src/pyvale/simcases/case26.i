#-------------------------------------------------------------------------
# pyvale: gmsh,mechanical,transient
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
#_* MOOSEHERDER VARIABLES - START

endTime = 80
timeStep = 1

# Mechanical Loads/BCs
topDispRate = ${fparse 1e-3 / endTime}  # m/s

# Mechanical Props: SS316L @ 20degC
ss316LEMod = 200e9       # Pa
ss316LPRatio = 0.3      # -

#** MOOSEHERDER VARIABLES - END
#-------------------------------------------------------------------------

[GlobalParams]
    displacements = 'disp_x disp_y disp_z'
[]

[Mesh]
    type = FileMesh
    file = 'case26.msh'
[]

[Modules/TensorMechanics/Master]
    [all]
        strain = SMALL
        incremental = true
        add_variables = true
        material_output_family = MONOMIAL   # MONOMIAL, LAGRANGE
        material_output_order = SECOND       # CONSTANT, FIRST, SECOND,
        generate_output = 'strain_xx strain_yy strain_zz strain_xy strain_yz strain_xz'
    []
[]

[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-base-disp'
        value = 0.0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = 'bc-base-disp'
        value = 0.0
    []
    [bottom_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-base-disp'
        value = 0.0
    []


    [top_x]
        type = DirichletBC
        variable = disp_x
        boundary = 'bc-top-disp'
        value = 0.0
    []
    [top_y]
        type = FunctionDirichletBC
        variable = disp_y
        boundary = 'bc-top-disp'
        function = '${topDispRate}*t'
    []
    [top_z]
        type = DirichletBC
        variable = disp_z
        boundary = 'bc-top-disp'
        value = 0.0
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${ss316LEMod}
        poissons_ratio = ${ss316LPRatio}
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
    solve_type = 'NEWTON'
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
        boundary = 'bc-base-disp'
    []
    [react_y_top]
        type = SidesetReaction
        direction = '0 1 0'
        stress_tensor = stress
        boundary = 'bc-top-disp'
    []

    [disp_y_max]
        type = NodalExtremeValue
        variable = disp_y
    []
    [disp_x_max]
        type = NodalExtremeValue
        variable = disp_x
    []
    [disp_z_max]
        type = NodalExtremeValue
        variable = disp_x
    []
[]

[Outputs]
    exodus = true
[]