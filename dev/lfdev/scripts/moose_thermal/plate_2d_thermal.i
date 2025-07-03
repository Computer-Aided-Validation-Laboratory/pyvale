#_*
end_time = 30
time_step = 0.5
max_temp = 500
init_temp = 20.0
#**

[Mesh]
    [generated]
        type = GeneratedMeshGenerator
        dim = 2
        nx = 20
        ny = 10
        xmax = 2
        ymax = 1
    []
[]

[Variables]
    [temperature]
        initial_condition = ${init_temp}
    []
[]

[Kernels]
    [heat_conduction]
        type = HeatConduction
        variable = temperature
    []
    [time_derivative]
        type = HeatConductionTimeDerivative
        variable = temperature
    []
    [heat_source]
        type = HeatSource
        variable = temperature
        value = 1e4
    []
[]

[Materials]
    [thermal]
        type = HeatConductionMaterial
        thermal_conductivity = 45.0
        specific_heat = 0.5
    []
    [density]
        type = GenericConstantMaterial
        prop_names = 'density'
        prop_values = 8000.0
    []
[]

[BCs]
    [t_left]
        type = DirichletBC
        variable = temperature
        value = ${init_temp}
        boundary = 'left'
    []
    [t_right]
        type = FunctionDirichletBC
        variable = temperature
        function = '${init_temp} + ${max_temp}*(1-exp(-t))'
        boundary = 'right'
    []
[]

[Executioner]
    type = Transient
    end_time = ${end_time}
    dt = ${time_step}
[]

[Outputs]
    exodus = true
[]