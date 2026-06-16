#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <stdexcept>

#include "../../../common_cpp/Eigen/Dense"

#include "meshio.h"

// SimData loadData(const std::string& path) 
// {
//     std::string coords_path = path + "/coords.csv";
//     std::string connect_path = path + "/connectivity.csv";

//     std::vector<std::string> field_paths{
//         path + "/field_disp_x.csv",
//         path + "/field_disp_y.csv",
//         path + "/field_disp_z.csv"
//     };

//     return loadSimData(
//         coords_path,
//         connect_path,
//         &field_paths,
//         nullptr
//     );
// };


SimData loadData(const std::string& path)
{
    namespace fs = std::filesystem;

    std::string coords_path   = path + "/coords.csv";
    std::string connect_path  = path + "/connectivity.csv";

    std::vector<std::string> field_paths{
        path + "/field_disp_x.csv",
        path + "/field_disp_y.csv",
        path + "/field_disp_z.csv"
    };

    auto check_file = [](const std::string& p)
    {
        if (!fs::exists(p))
        {
            throw std::runtime_error("Missing file: " + p);
        }
        if (!fs::is_regular_file(p))
        {
            throw std::runtime_error("Not a regular file: " + p);
        }
    };

    check_file(coords_path);
    check_file(connect_path);

    for (const auto& p : field_paths)
    {
        check_file(p);
    }

    return loadSimData(
        coords_path,
        connect_path,
        &field_paths,
        nullptr
    );
}

