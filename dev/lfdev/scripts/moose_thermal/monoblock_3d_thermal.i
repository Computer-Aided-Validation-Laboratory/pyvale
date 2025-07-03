#-------------------------------------------------------------------------
# monoblock_transient_thermal.i
# Author: Luke Humphrey
# (c) Copyright UKAEA 2024.
#
#-------------------------------------------------------------------------
# DESCRIPTION
#
# Input file for a thermal simulation of a divertor monoblock.
#
# The monoblock is comprised of a copper-chromium-zirconium (CuCrZr) pipe
# surrounded by tungsten armour with an OFHC copper pipe interlayer in between.
#
# Temperature-variant material properties are implemented via linear
# interpolation from available data. Some of these material properties are not
# used for this thermal simulation, but are in place for a thermomechanical
# model including thermal expansion.
#
# Parameters describing the geometry are present at the top of the file above
# the MOOSE tree structure. These parameters can be modified to produce a
# monoblock design with the specified geometry.
#
# The mesh uses first order elements with a nominal mesh refinement of one
# division per millimetre.
#
# The incoming heat is modelled as a constant heat flux on the top surface of
# the block (i.e. the plasma-facing side). The outgoing heat is modelled as a
# convective heat flux on the internal surface of the copper pipe. Besides this
# heat flux, coolant flow is not modelled; the fluid region is treated as void.
#
# The boundary conditions are the incoming heat flux on the top surface, and
# the coolant temperature. The solve is steady state and outputs temperature.

#-------------------------------------------------------------------------
# PARAMETER DEFINITIONS

end_time = 30
time_step = 0.5

#_*

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Geometry
PI=3.141592653589793

pipeThick=1.5e-3     # m
pipeIntDiam=12e-3    # m
pipeExtDiam=${fparse pipeIntDiam + 2*pipeThick}

intLayerThick=1e-3   # m
intLayerIntDiam=${pipeExtDiam}
intLayerExtDiam=${fparse intLayerIntDiam + 2*intLayerThick}

monoBThick=3e-3      # m
monoBWidth=${fparse intLayerExtDiam + 2*monoBThick}
monoBArmHeight=8e-3  # m
monoBDepth=12e-3     # m

pipeIntCirc=${fparse PI * pipeIntDiam}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Mesh Sizing
meshRefFact=1
meshDens=1e3 # divisions per metre (nominal)

# Mesh Order
secondOrder=false
orderString=FIRST

# Note: some of the following values must be even integers. This is in some
# cases a requirement for the meshing functions used, else is is to ensure a
# division is present at the centreline, thus allowing zero-displacement
# boundary conditions to be applied to the centre nodes. These values are
# halved, rounded to int, then doubled to ensure the result is an even int.

# Number of divisions along the top section of the monoblock armour.
monoBArmDivs=${fparse int(monoBArmHeight * meshDens * meshRefFact)}

# Number of divisions around each quadrant of the circumference of the pipe,
# interlayer, and radial section of the monoblock armour.
pipeCircSectDivs=${fparse 2 * int(monoBWidth/2 * meshDens * meshRefFact / 2)}

# Number of radial divisions for the pipe, interlayer, and radial section of
# the monoblock armour respectively.
pipeRadDivs=${fparse max(int(pipeThick * meshDens * meshRefFact), 3)}
intLayerRadDivs=${fparse max(int(intLayerThick * meshDens * meshRefFact), 5)}
monoBRadDivs=${
  fparse max(int((monoBWidth-intLayerExtDiam)/2 * meshDens * meshRefFact), 5)
}

# Number of divisions along monoblock depth (i.e. z-dimension).
extrudeDivs=${fparse max(2 * int(monoBDepth * meshDens * meshRefFact / 2), 4)}

monoBElemSize=${fparse monoBDepth / extrudeDivs}
tol=${fparse monoBElemSize / 10}
ctol=${fparse pipeIntCirc / (8 * 4 * pipeCircSectDivs)}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Material Properties
# Mono-Block/Armour = Tungsten
# Interlayer = Oxygen-Free High-Conductivity (OFHC) Copper
# Cooling pipe = Copper Chromium Zirconium (CuCrZr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Loads and BCs
stressFreeTemp=20   # degC
coolantTemp=150     # degC
surfHeatFlux=10e6   # W/m^2

#**
#-------------------------------------------------------------------------

[Mesh]
  second_order = ${secondOrder}

  [mesh_monoblock]
    type = PolygonConcentricCircleMeshGenerator
    num_sides = 4
    polygon_size = ${fparse monoBWidth / 2}
    polygon_size_style = apothem  # i.e. distance from centre to edge
    ring_radii = '
      ${fparse pipeIntDiam / 2}
      ${fparse pipeExtDiam / 2}
      ${fparse intLayerExtDiam / 2}
    '
    num_sectors_per_side = '
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
      ${pipeCircSectDivs}
    '
    ring_intervals = '1 ${pipeRadDivs} ${intLayerRadDivs}'
    background_intervals = ${monoBRadDivs}
    preserve_volumes = on
    flat_side_up = true
    ring_block_names = 'void pipe interlayer'
    background_block_names = monoblock
    interface_boundary_id_shift = 1000
    external_boundary_name = monoblock_boundary
    generate_side_specific_boundaries = true
  []

  [mesh_armour]
    type = GeneratedMeshGenerator
    dim = 2
    xmin = ${fparse monoBWidth /-2}
    xmax = ${fparse monoBWidth / 2}
    ymin = ${fparse monoBWidth / 2}
    ymax = ${fparse monoBWidth / 2 + monoBArmHeight}
    nx = ${pipeCircSectDivs}
    ny = ${monoBArmDivs}
    boundary_name_prefix = armour
  []

  [combine_meshes]
    type = StitchedMeshGenerator
    inputs = 'mesh_monoblock mesh_armour'
    stitch_boundaries_pairs = 'monoblock_boundary armour_bottom'
    clear_stitched_boundary_ids = true
  []

  [delete_void]
    type = BlockDeletionGenerator
    input = combine_meshes
    block = void
    new_boundary = internal_boundary
  []

  [merge_block_names]
    type = RenameBlockGenerator
    input = delete_void
    old_block = '4 0'
    new_block = 'armour armour'
  []

  [merge_boundary_names]
    type = RenameBoundaryGenerator
    input = merge_block_names
    old_boundary = 'armour_top
                    armour_left 10002 15002
                    armour_right 10004 15004
                    10003 15003'
    new_boundary = 'top
                    left left left
                    right right right
                    bottom bottom'
  []

  [extrude]
    type = AdvancedExtruderGenerator
    input = merge_boundary_names
    direction = '0 0 1'
    heights = ${monoBDepth}
    num_layers = ${extrudeDivs}
  []

  [name_node_centre_x_bottom_y_back_z]
    type = BoundingBoxNodeSetGenerator
    input = extrude
    bottom_left = '${fparse -ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse -tol}'
    top_right = '${fparse ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse tol}'
    new_boundary = centre_x_bottom_y_back_z
  []
  [name_node_centre_x_bottom_y_front_z]
    type = BoundingBoxNodeSetGenerator
    input = name_node_centre_x_bottom_y_back_z
    bottom_left = '${fparse -ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse monoBDepth-tol}'
    top_right = '${fparse ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse monoBDepth+tol}'
    new_boundary = centre_x_bottom_y_front_z
  []
  [name_node_left_x_bottom_y_centre_z]
    type = BoundingBoxNodeSetGenerator
    input = name_node_centre_x_bottom_y_front_z
    bottom_left = '${fparse (monoBWidth/-2)-ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse (monoBDepth/2)-tol}'
    top_right = '${fparse (monoBWidth/-2)+ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse (monoBDepth/2)+tol}'
    new_boundary = left_x_bottom_y_centre_z
  []
  [name_node_right_x_bottom_y_centre_z]
    type = BoundingBoxNodeSetGenerator
    input = name_node_left_x_bottom_y_centre_z
    bottom_left = '${fparse (monoBWidth/2)-ctol}
                   ${fparse (monoBWidth/-2)-ctol}
                   ${fparse (monoBDepth/2)-tol}'
    top_right = '${fparse (monoBWidth/2)+ctol}
                 ${fparse (monoBWidth/-2)+ctol}
                 ${fparse (monoBDepth/2)+tol}'
    new_boundary = right_x_bottom_y_centre_z
  []
[]

[Variables]
  [temperature]
    family = LAGRANGE
    order = ${orderString}
    initial_condition = ${coolantTemp}
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
[]

[Functions]
  [cucrzr_thermal_expansion]
    type = PiecewiseLinear
    xy_data = '
      20 1.67e-05
      50 1.7e-05
      100 1.73e-05
      150 1.75e-05
      200 1.77e-05
      250 1.78e-05
      300 1.8e-05
      350 1.8e-05
      400 1.81e-05
      450 1.82e-05
      500 1.84e-05
      550 1.85e-05
      600 1.86e-05
    '
  []
  [copper_thermal_expansion]
    type = PiecewiseLinear
    xy_data = '
      20 1.67e-05
      50 1.7e-05
      100 1.72e-05
      150 1.75e-05
      200 1.77e-05
      250 1.78e-05
      300 1.8e-05
      350 1.81e-05
      400 1.82e-05
      450 1.84e-05
      500 1.85e-05
      550 1.87e-05
      600 1.88e-05
      650 1.9e-05
      700 1.91e-05
      750 1.93e-05
      800 1.96e-05
      850 1.98e-05
      900 2.01e-05
    '
  []
  [tungsten_thermal_expansion]
    type = PiecewiseLinear
    xy_data = '
      20 4.5e-06
      100 4.5e-06
      200 4.53e-06
      300 4.58e-06
      400 4.63e-06
      500 4.68e-06
      600 4.72e-06
      700 4.76e-06
      800 4.81e-06
      900 4.85e-06
      1000 4.89e-06
      1200 4.98e-06
      1400 5.08e-06
      1600 5.18e-06
      1800 5.3e-06
      2000 5.43e-06
      2200 5.57e-06
      2400 5.74e-06
      2600 5.93e-06
      2800 6.15e-06
      3000 6.4e-06
      3200 6.67e-06
    '
  []
[]

[Materials]
  [cucrzr_thermal_conductivity]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 318
      50 324
      100 333
      150 339
      200 343
      250 345
      300 346
      350 347
      400 347
      450 346
      500 346
    '
    variable = temperature
    property = thermal_conductivity
    block = 'pipe'
  []
  [copper_thermal_conductivity]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 401
      50 398
      100 395
      150 391
      200 388
      250 384
      300 381
      350 378
      400 374
      450 371
      500 367
      550 364
      600 360
      650 357
      700 354
      750 350
      800 347
      850 344
      900 340
      950 337
      1000 334
    '
    variable = temperature
    property = thermal_conductivity
    block = 'interlayer'
  []
  [tungsten_thermal_conductivity]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 173
      50 170
      100 165
      150 160
      200 156
      250 151
      300 147
      350 143
      400 140
      450 136
      500 133
      550 130
      600 127
      650 125
      700 122
      750 120
      800 118
      850 116
      900 114
      950 112
      1000 110
      1100 108
      1200 105
    '
    variable = temperature
    property = thermal_conductivity
    block = 'armour'
  []

  [cucrzr_density]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 8900
      50 8886
      100 8863
      150 8840
      200 8816
      250 8791
      300 8797
      350 8742
      400 8716
      450 8691
      500 8665
    '
    variable = temperature
    property = density
    block = 'pipe'
  []
  [copper_density]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 8940
      50 8926
      100 8903
      150 8879
      200 8854
      250 8829
      300 8802
      350 8774
      400 8744
      450 8713
      500 8681
      550 8647
      600 8612
      650 8575
      700 8536
      750 8495
      800 8453
      850 8409
      900 8363
    '
    variable = temperature
    property = density
    block = 'interlayer'
  []
  [tungsten_density]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 19300
      50 19290
      100 19280
      150 19270
      200 19250
      250 19240
      300 19230
      350 19220
      400 19200
      450 19190
      500 19180
      550 19170
      600 19150
      650 19140
      700 19130
      750 19110
      800 19100
      850 19080
      900 19070
      950 19060
      1000 19040
      1100 19010
      1200 18990
    '
    variable = temperature
    property = density
    block = 'armour'
  []

  [cucrzr_elastic_modulus]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 128000000000.0
      50 127000000000.0
      100 127000000000.0
      150 125000000000.0
      200 123000000000.0
      250 121000000000.0
      300 118000000000.0
      350 116000000000.0
      400 113000000000.0
      450 110000000000.0
      500 106000000000.0
      550 100000000000.0
      600 95000000000.0
      650 90000000000.0
      700 86000000000.0
    '
    variable = temperature
    property = elastic_modulus
    block = 'pipe'
  []
  [copper_elastic_modulus]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 117000000000.0
      50 116000000000.0
      100 114000000000.0
      150 112000000000.0
      200 110000000000.0
      250 108000000000.0
      300 105000000000.0
      350 102000000000.0
      400 98000000000.0
    '
    variable = temperature
    property = elastic_modulus
    block = 'interlayer'
  []
  [tungsten_elastic_modulus]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 398000000000.0
      50 398000000000.0
      100 397000000000.0
      150 397000000000.0
      200 396000000000.0
      250 396000000000.0
      300 395000000000.0
      350 394000000000.0
      400 393000000000.0
      450 391000000000.0
      500 390000000000.0
      550 388000000000.0
      600 387000000000.0
      650 385000000000.0
      700 383000000000.0
      750 381000000000.0
      800 379000000000.0
      850 376000000000.0
      900 374000000000.0
      950 371000000000.0
      1000 368000000000.0
      1100 362000000000.0
      1200 356000000000.0
    '
    variable = temperature
    property = elastic_modulus
    block = 'armour'
  []

  [cucrzr_specific_heat]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 390
      50 393
      100 398
      150 402
      200 407
      250 412
      300 417
      350 422
      400 427
      450 432
      500 437
      550 442
      600 447
      650 452
      700 458
    '
    variable = temperature
    property = specific_heat
    block = 'pipe'
  []
  [copper_specific_heat]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 388
      50 390
      100 394
      150 398
      200 401
      250 406
      300 410
      350 415
      400 419
      450 424
      500 430
      550 435
      600 441
      650 447
      700 453
      750 459
      800 466
      850 472
      900 479
      950 487
      1000 494
    '
    variable = temperature
    property = specific_heat
    block = 'interlayer'
  []
  [tungsten_specific_heat]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      20 129
      50 130
      100 132
      150 133
      200 135
      250 136
      300 138
      350 139
      400 141
      450 142
      500 144
      550 145
      600 147
      650 148
      700 150
      750 151
      800 152
      850 154
      900 155
      950 156
      1000 158
      1100 160
      1200 163
    '
    variable = temperature
    property = specific_heat
    block = 'armour'
  []

  [cucrzr_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.33
    block = 'pipe'
  []
  [copper_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.33
    block = 'interlayer'
  []
  [tungsten_elasticity]
    type = ComputeVariableIsotropicElasticityTensor
    args = temperature
    youngs_modulus = elastic_modulus
    poissons_ratio = 0.29
    block = 'armour'
  []

  [cucrzr_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = cucrzr_thermal_expansion
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'pipe'
  []
  [copper_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = copper_thermal_expansion
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'interlayer'
  []
  [tungsten_expansion]
    type = ComputeInstantaneousThermalExpansionFunctionEigenstrain
    temperature = temperature
    stress_free_temperature = ${stressFreeTemp}
    thermal_expansion_function = tungsten_thermal_expansion
    eigenstrain_name = thermal_expansion_eigenstrain
    block = 'armour'
  []

  [coolant_heat_transfer_coefficient]
    type = PiecewiseLinearInterpolationMaterial
    xy_data = '
      1 4
      100 109.1e3
      150 115.9e3
      200 121.01e3
      250 128.8e3
      295 208.2e3
    '
    variable = temperature
    property = heat_transfer_coefficient
    boundary = 'internal_boundary'
  []
[]

[BCs]
  [heat_flux_in]
    type = FunctionNeumannBC
    variable = temperature
    boundary = 'top'
    function = '${fparse surfHeatFlux}*(1-exp(-t))'
  []
  [heat_flux_out]
    type = ConvectiveHeatFluxBC
    variable = temperature
    boundary = 'internal_boundary'
    T_infinity = ${coolantTemp}
    heat_transfer_coefficient = heat_transfer_coefficient
  []
[]

[Preconditioning]
  [smp]
    type = SMP
    full = true
  []
[]

[Executioner]
  type = Transient
  solve_type = 'PJFNK'
  petsc_options_iname = '-pc_type -pc_hypre_type'
  petsc_options_value = 'hypre boomeramg'
  end_time = ${end_time}
  dt = ${time_step}
[]

[Postprocessors]
  [max_temp]
    type = ElementExtremeValue
    variable = temperature
  []
[]

[Outputs]
  exodus = true
[]
