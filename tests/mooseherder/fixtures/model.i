!include parameters.i
nElemX = 2
nElemY = 2
[Mesh/generated]
  type = GeneratedMeshGenerator
  dim = 2
  nx = ${nElemX}
  ny = ${nElemY}
  xmax = 1
  ymax = 1
  elem_type = QUAD4
[]
!include physics.i
!include solver_smp.i
