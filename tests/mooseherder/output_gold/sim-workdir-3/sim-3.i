# This is a simple MOOSE tensor mechanics input script for testing the herder
dimension = 2 # Putting this variable outside the block to test

#_* Variables Block
n_elem_x = 20
n_elem_y = 10 # Testing comments in the variables block
e_modulus = 2000000000.0
# Comment line to test
p_ratio = 0.3 # Another comment to test with 
e_type = QUAD4
time_end = 3
#**

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = ${dimension}
        nx = ${n_elem_x}
        ny = ${n_elem_y}
        xmax = 2
        ymax = 1
        elem_type = ${e_type}
    []
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = true
        generate_output = 'vonmises_stress strain_xx strain_yy strain_xy strain_zz'
    []
[]

#
# Added boundary/loading conditions
# https://mooseframework.inl.gov/modules/tensor_mechanics/tutorials/introduction/step02.html
#
[BCs]
    [bottom_x]
        type = DirichletBC
        variable = disp_x
        boundary = bottom
        value = 0
    []
    [bottom_y]
        type = DirichletBC
        variable = disp_y
        boundary = bottom
        value = 0
    []
    [Pressure]
        [top]
        boundary = top
        function = 1e7*t
        []
    []
[]

[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${e_modulus}
        poissons_ratio = ${p_ratio}
    []
    [stress]
        type = ComputeLinearElasticStress
    []
[]

# consider all off-diagonal Jacobians for preconditioning
[Preconditioning]
    [SMP]
        type = SMP
        full = true
    []
[]

[Executioner]
    type = Transient
    # we chose a direct solver here
    petsc_options_iname = '-pc_type'
    petsc_options_value = 'lu'
    end_time = ${time_end}
    dt = 1
[]

[Outputs]
    exodus = true
[]