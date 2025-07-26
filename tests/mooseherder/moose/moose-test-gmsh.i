#_* Variables
e_modulus = 1e9
p_ratio = 0.3
#**

[GlobalParams]
    displacements = 'disp_x disp_y'
[]

[Mesh]
    type = FileMesh
    file = 'gmsh-test.msh'
[]

[Modules/TensorMechanics/Master]
    [all]
        add_variables = true
        generate_output = 'vonmises_stress strain_xx strain_yy strain_zz'
    []
[]
  
[BCs]
    [./u_top_pull]
        type = Pressure
        variable = disp_y
        boundary = Top-BC
        function = -1e7*t
    [../]
    [./u_bottom_fix]
        type = DirichletBC
        variable = disp_y
        boundary = Btm-BC
        value = 0.0
    [../]
    [./u_yz_fix]
        type = DirichletBC
        variable = disp_x
        boundary = Mid-BC
        value = 0.0
    [../]
[]
  
[Materials]
    [elasticity]
        type = ComputeIsotropicElasticityTensor
        youngs_modulus = ${e_modulus}
        poissons_ratio = ${p_ratio}
    []
    [stress]
        type = ComputeLinearElasticStress
        #type = ComputeFiniteStrainElasticStress
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
    petsc_options_iname = '-pc_type'
    petsc_options_value = 'lu'
    end_time = 2
    dt = 1
[]
  

[Outputs]
    exodus = true
[]