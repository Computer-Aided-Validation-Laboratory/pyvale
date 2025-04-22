// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>

// Program Header files




namespace smooth {

    std::vector<double> bilinear(const std::vector<double>& field, int px_horizontal, int px_vertical);
    std::vector<double> biquadratic(const std::vector<double>& field, int px_horizontal, int px_vertical);


}
