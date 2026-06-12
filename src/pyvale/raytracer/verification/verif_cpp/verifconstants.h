#pragma once

#include <iostream>
#include <limits>
#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <optional>
#include <cstddef>
#include <cstddef>

#include "../../../common_cpp/Eigen/Dense"

using Vec3f = Eigen::Vector3d;
using MatXf = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <std::size_t N>
using NodeArray = std::array<EiVector3d, N>;

struct DistortCase
{
    std::string case_name;
    enum ElementNodeCount nodes_per_element;
    std::string data_dir;
    EiVector3d camera_center;
};

std::vector<DistortCase> distort_cases = {
    // {
    //     "bulge",
    //     ElementNodeCount::TRI6,
    //     "data/edge/tri6_distort_bulge",
    //     EiVector3d(5.0, 2.8301270189221928, 242.00527248356275)
    // },
    // {
    //     "tan",
    //     ElementNodeCount::TRI6,
    //     "data/edge/tri6_distort_tan",
    //     EiVector3d(5.0, 4.3301270189221930, 179.74112154016652)
    // },
    // {
    //     "stretch",
    //     ElementNodeCount::TRI6,
    //     "data/edge/tri6_distort_stretch",
    //     EiVector3d(59.33012701892219, 0.0, 1539.2249934154347)
    // },
    // {
    //     "shear",
    //     ElementNodeCount::TRI6,
    //     "data/edge/tri6_distort_shear",
    //     EiVector3d(57.5, 4.330127018922193, 1491.7452830188683)
    // },
    // {
    //     "rot",
    //     ElementNodeCount::TRI6,
    //     "data/edge/tri6_distort_rot",
    //     EiVector3d(5.0, 2.886751345948129, 144.92861657761585)
    // },
    // {
    //     "bulge",
    //     ElementNodeCount::QUAD8,
    //     "data/edge/quad8_distort_bulge",
    //     EiVector3d(5.0, 5.0, 332.07547169811323)
    // },
    // {
    //     "tan",
    //     ElementNodeCount::QUAD8,
    //     "data/edge/quad8_distort_tan",
    //     EiVector3d(5.0, 5.0, 207.54716981132077)
    // },
    // {
    //     "stretch",
    //     ElementNodeCount::QUAD8,
    //     "data/edge/quad8_distort_stretch",
    //     EiVector3d(60.0, 5.0, 1556.6037735849059)
    // },
    // {
    //     "shear",
    //     ElementNodeCount::QUAD8,
    //     "data/edge/quad8_distort_shear",
    //     EiVector3d(60.0, 5.0, 1556.6037735849059)
    // },
    // {
    //     "rot",
    //     ElementNodeCount::QUAD8,
    //     "data/edge/quad8_distort_rot",
    //     EiVector3d(5.0, 5.0, 167.27358490566039)
    // },
    // {
    //     "bulge",
    //     ElementNodeCount::QUAD9,
    //     "data/edge/quad9_distort_bulge",
    //     EiVector3d(5.0, 5.0, 332.07547169811323)
    // },
    {
        "shear",
        ElementNodeCount::QUAD9,
        "data/edge/quad9_distort_shear",
        EiVector3d(60.0, 5.0, 1556.6037735849059)
    },
};