// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once

#include <vector>
#include <cstdlib>
#include <string>
#include <cstdint>

// common_cpp header files
#include "./util.hpp"

Image  read_img(const std::string& filename);
Image  read_bmp(const std::string& filename);
Image read_tiff(const std::string& filename);

