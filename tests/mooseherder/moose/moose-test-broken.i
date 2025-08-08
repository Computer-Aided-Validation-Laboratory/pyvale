# This is a simple MOOSE tensor mechanics input script for testing the herder
n_elem_x = 40 # Putting this variable outside the block to test

#_* Variables Block
# n_elem_x = 50 putting an equals sign here to test
# n_elem_y = 20 # Testing comments in the variables block, this line is broken
e_modulus = 1e9
# Comment line to test
p_ratio = 0.3# Another comment to test with
# The next variables test strings
e_type = QUAD4
add_vars = true
y_max = 1 
x_max = ${fparse 2*y_max}
#**

# Another variable outside the block to test
spatial_dims = 2

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = ${spatial_dims}
        nx = ${n_elem_x}
        ny = ${n_elem_y}
        xmax = ${x_max}
        ymax = ${y_max}
        elem_type = ${e_type}
    []
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = ${add_vars}
        generate_output = 'vonmises_stress strain_xx strain_yy strain_xy strain_zz'
    []
[]

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
    end_time = 5
    dt = 1
[]

[Outputs]
    exodus = true
[]