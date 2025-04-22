// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <vector>

// Program Header files

namespace smooth {

    std::vector<double> bilinear(const std::vector<double> &field,
                                 int px_horizontal, int px_vertical) {

        std::vector<double> result(field.size(), 0.0);
    
        for (int y = 0; y < px_vertical; ++y) {
            for (int x = 0; x < px_horizontal; ++x) {
                double sum = 0.0;
                int count = 0;
    
                // Q4: 2x2 stencil (bilinear - immediate neighbors)
                for (int j = 0; j <= 1; ++j) {
                    for (int i = 0; i <= 1; ++i) {
                        sum += field[(y+j)*px_horizontal + (x+i)];
                        ++count;
                    }
                }
    
                result[y*px_horizontal + x] = sum / count;
            }
        }
    
        return result;
    }
    
    std::vector<double> biquadratic(const std::vector<double> &field,
                                    int px_horizontal, int px_vertical) {
    
        std::vector<double> result(field.size(), 0.0);
    
        for (int y = 0; y < px_vertical; ++y) {
            for (int x = 0; x < px_horizontal; ++x) {
                double sum = 0.0;
                int count = 0;
    
                // Q9: 3x3 stencil (biquadratic - 9-point average)
                for (int j = -1; j <= 1; ++j) {
                    for (int i = -1; i <= 1; ++i) {
                        sum += field[(y+j)*px_horizontal + (x+i)];
                        ++count;
                    }
                }
    
                result[y*px_horizontal + x] = sum / count;
            }
        }
    
        return result;
    }

} // namespace smooth
