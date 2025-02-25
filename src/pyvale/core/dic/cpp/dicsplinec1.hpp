// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef SPLINEC1_H
#define SPLINEC1_H


#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>

namespace splinec1 {


    // Utility function to access the grid as if it's a 2D array
    inline double& at(std::vector<double>& grid, int px_vertical, int px_horizontal, int x, int y) {
        return grid[y * px_horizontal + x];
    }

    // Cubic interpolation kernel
    double catmullrom(double p0, double p1, double p2, double p3, double t) {
        double a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
        double b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3;
        double c = -0.5 * p0 + 0.5 * p2;
        double d = p1;

        return a * t * t * t + b * t * t + c * t + d;
    }


    double bicubicInterpolateCatmullRom(std::vector<double>& grid, int px_vertical, int px_horizontal, double x, double y) {


        int x1 = std::floor(x);
        int y1 = std::floor(y);
        double dx = x - x1;
        double dy = y - y1;


        // 4x4 block of neighbors
        double p[4][4];
        for (int yy = -1; yy < 3; yy++) {
            for (int xx = -1; xx < 3; xx++) {
                
                int gridX = std::clamp(x1 + xx, 0, px_horizontal - 1);
                int gridY = std::clamp(y1 + yy, 0, px_vertical - 1);

                p[yy+1][xx+1] = at(grid, px_vertical, px_horizontal, gridX, gridY);
                // std::cout << p[yy][xx] << std::endl;


            }
            // std::cout << std::endl;
        }

        double row[4];
        for (int i = 0; i < 4; ++i) {
            row[i] = catmullrom(p[i][0], p[i][1], p[i][2], p[i][3], dx);
        }

        // Interpolate in the y-direction
        return catmullrom(row[0], row[1], row[2], row[3], dy);
    }

}

#endif /* TK_SPLINE_H */