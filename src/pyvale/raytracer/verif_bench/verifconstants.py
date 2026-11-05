from pyvale.raytracer.rtmesh import *

distort_cases = [
    {
        "case_name": "bulge",
        "mesh_type": ElementNodeCount.TRI6,
        "data_dir": "data/edge/tri6_distort_bulge",
        "camera_input": {
            "pos_world": (5.0, 2.8301270189221928, 242.00527248356275),
            "roi_cent_world": (5.0, 2.8301270189221928, 0.0),
        },
    },
    {
        "case_name": "bulge",
        "mesh_type": ElementNodeCount.QUAD8,
        "data_dir": "data/edge/quad8_distort_bulge",
        "camera_input": {
            "pos_world": (5.0, 5.0, 332.07547169811323),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "bulge",
        "mesh_type": ElementNodeCount.QUAD9,
        "data_dir": "data/edge/quad9_distort_bulge",
        "camera_input": {
            "pos_world": (5.0, 5.0, 332.07547169811323),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "tan",
        "mesh_type": ElementNodeCount.TRI6,
        "data_dir": "data/edge/tri6_distort_tan",
        "camera_input": {
            "pos_world": (5.0, 4.3301270189221930, 179.74112154016652),
            "roi_cent_world": (5.0, 4.3301270189221930, 0.0),
        },
    },
    {
        "case_name": "tan",
        "mesh_type": ElementNodeCount.QUAD8,
        "data_dir": "data/edge/quad8_distort_tan",
        "camera_input": {
            "pos_world": (5.0, 5.0, 207.54716981132077),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "tan",
        "mesh_type": ElementNodeCount.QUAD9,
        "data_dir": "data/edge/quad9_distort_tan",
        "camera_input": {
            "pos_world": (5.0, 5.0, 207.54716981132077),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "stretch",
        "mesh_type": ElementNodeCount.TRI3,
        "data_dir": "data/edge/tri3_distort_stretch",
        "camera_input": {
            "pos_world": (59.33012701892219, 0.0, 1539.2249934154347),
            "roi_cent_world": (59.33012701892219, 0.0, 0.0),
        },
    },
    {
        "case_name": "stretch",
        "mesh_type": ElementNodeCount.TRI6,
        "data_dir": "data/edge/tri6_distort_stretch",
        "camera_input": {
            "pos_world": (59.33012701892219, 0.0, 1539.2249934154347),
            "roi_cent_world": (59.33012701892219, 0.0, 0.0),
        },
    },
    {
        "case_name": "stretch",
        "mesh_type": ElementNodeCount.QUAD4,
        "data_dir": "data/edge/quad4_distort_stretch",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "stretch",
        "mesh_type": ElementNodeCount.QUAD8,
        "data_dir": "data/edge/quad8_distort_stretch",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "stretch",
        "mesh_type": ElementNodeCount.QUAD9,
        "data_dir": "data/edge/quad9_distort_stretch",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "shear",
        "mesh_type": ElementNodeCount.TRI3,
        "data_dir": "data/edge/tri3_distort_shear",
        "camera_input": {
            "pos_world": (57.5, 4.330127018922193, 1491.7452830188683),
            "roi_cent_world": (57.5, 4.330127018922193, 0.0),
        },
    },
    {
        "case_name": "shear",
        "mesh_type": ElementNodeCount.TRI6,
        "data_dir": "data/edge/tri6_distort_shear",
        "camera_input": {
            "pos_world": (57.5, 4.330127018922193, 1491.7452830188683),
            "roi_cent_world": (57.5, 4.330127018922193, 0.0),
        },
    },
    {
        "case_name": "shear",
        "mesh_type": ElementNodeCount.QUAD4,
        "data_dir": "data/edge/quad4_distort_shear",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "shear",
        "mesh_type": ElementNodeCount.QUAD8,
        "data_dir": "data/edge/quad8_distort_shear",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "shear",
        "mesh_type": ElementNodeCount.QUAD9,
        "data_dir": "data/edge/quad9_distort_shear",
        "camera_input": {
            "pos_world": (60.0, 5.0, 1556.6037735849059),
            "roi_cent_world": (60.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "rot",
        "mesh_type": ElementNodeCount.TRI3,
        "data_dir": "data/edge/tri3_distort_rot",
        "camera_input": {
            "pos_world": (5.0, 2.886751345948129, 144.92861657761585),
            "roi_cent_world": (5.0, 2.886751345948129, 0.0),
        },
    },
    {
        "case_name": "rot",
        "mesh_type": ElementNodeCount.TRI6,
        "data_dir": "data/edge/tri6_distort_rot",
        "camera_input": {
            "pos_world": (5.0, 2.886751345948129, 144.92861657761585),
            "roi_cent_world": (5.0, 2.886751345948129, 0.0),
        },
    },
    {
        "case_name": "rot",
        "mesh_type": ElementNodeCount.QUAD4,
        "data_dir": "data/edge/quad4_distort_rot",
        "camera_input": {
            "pos_world": (5.0, 5.0, 167.27358490566039),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "rot",
        "mesh_type": ElementNodeCount.QUAD8,
        "data_dir": "data/edge/quad8_distort_rot",
        "camera_input": {
            "pos_world": (5.0, 5.0, 167.27358490566039),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
    {
        "case_name": "rot",
        "mesh_type": ElementNodeCount.QUAD9,
        "data_dir": "data/edge/quad9_distort_rot",
        "camera_input": {
            "pos_world": (5.0, 5.0, 167.27358490566039),
            "roi_cent_world": (5.0, 5.0, 0.0),
        },
    },
]