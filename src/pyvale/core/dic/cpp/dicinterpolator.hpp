// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================



#include <vector>
#include "./dicsplinec2.hpp" 

class BicubicInterpolator {


private:

    int px_horizontal, px_vertical;
    std::vector<tk::spline> row_splines;
    std::vector<double> X_row;  // X values for vertical interpolation

public:
    // Constructor: Precompute row-wise splines
    BicubicInterpolator(const std::vector<double>& grid, int width, int height)
        : px_horizontal(width), px_vertical(height), row_splines(width), X_row(width) {
        
        std::vector<double> X_col(height);
        std::vector<double> Y_col(height);

        // Initialize X_col
        for (int col = 0; col < height; col++) {
            X_col[col] = col;
        }
        
        // Compute splines for each row
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                Y_col[col] = grid[row * width + col];  // Row-major order
            }
            row_splines[row].set_points(X_col, Y_col, tk::spline::cspline);
            X_row[row] = row;
        }
    }

    // Operator () for querying interpolated values at (x, y)
    double operator()(double x, double y) const {
        std::vector<double> Y_interp(px_horizontal);
        
        // Interpolate along rows first
        for (int row = 0; row < px_horizontal; row++) {
            Y_interp[row] = row_splines[row](y);
        }

        // Interpolate along columns
        tk::spline s_vert;
        s_vert.set_points(X_row, Y_interp, tk::spline::cspline);

        return s_vert(x);
    }
};