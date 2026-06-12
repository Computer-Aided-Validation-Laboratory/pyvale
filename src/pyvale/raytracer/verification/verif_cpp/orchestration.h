#pragma once

#include <string>
#include <vector>

#include "../../../common_cpp/Eigen/Dense"

#include "meshio.h"

SimData loadData(const std::string& path) 
{
    std::string coords_path = path + "/coords.csv";
    std::string connect_path = path + "/connectivity.csv";

    std::vector<std::string> field_paths{
        path + "/field_disp_x.csv",
        path + "/field_disp_y.csv",
        path + "/field_disp_z.csv"
    };

    return loadSimData(
        coords_path,
        connect_path,
        &field_paths,
        nullptr
    );
};

