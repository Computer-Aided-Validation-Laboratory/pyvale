#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <sstream>

// custom header files
#include "../cpp/dicsplinec1.hpp"
#include "../cpp/dicsplinec2.hpp"



int main() {
    

    // -------------------------------------------------------------------------------------------
    // init our 2d image
    // -------------------------------------------------------------------------------------------
    int px_vertical = 300;
    int px_horizontal = 300;
    std::vector<double> grid(px_vertical * px_vertical, 0.0);

    // -------------------------------------------------------------------------------------------
    // output files
    // -------------------------------------------------------------------------------------------
    std::ofstream splinec2;
    std::stringstream sstr;
    sstr << "splinec2_output.dat";
    splinec2.open(sstr.str());


    // -------------------------------------------------------------------------------------------
    // Read the file line by line
    // -------------------------------------------------------------------------------------------
    std::string filename = "/home/kc4736/ukaea/pyvale/src/pyvale/examples/speckle/file.txt" ;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    
    std::string line;
    int row = 0;

    while (std::getline(file, line) && row < px_vertical) {
        std::istringstream stream(line);
        double value;
        int col = 0;

        // Parse each value in the line
        while (stream >> value && col < px_horizontal) {
            grid[row * px_horizontal + col] = value;
            col++;
        }

        row++;
    }

    std::vector<double> X_row(px_horizontal);
    std::vector<double> Y_row(px_horizontal);
    std::vector<double> X_col(px_vertical);
    std::vector<double> Y_col(px_vertical);
    std::vector<std::vector<double>> result(299 * 4, std::vector<double>(299 * 4));
    std::vector<tk::spline> s(px_vertical);
    tk::spline s_vert;


   // Precompute X values
    for (int col = 0; col < px_vertical; col++) X_col[col] = col;
    for (int row = 0; row < px_horizontal; row++) X_row[row] = row;

    // Compute row-wise splines
    for (int row = 0; row < px_horizontal; row++) {
        for (int col = 0; col < px_vertical; col++) {
            Y_col[col] = grid[row * px_horizontal + col];  // Row-major access
        }
        s[row].set_points(X_col, Y_col, tk::spline::cspline_hermite);
    }

    // Compute interpolation at non-grid points
    for (int i = 0; i < 299 * 4 - 1; i++) {
        double x = i * 0.25;

        // Compute vertical interpolation values for each row at y
        for (int row = 0; row < px_horizontal; row++) {
            Y_row[row] = s[row](x);
        }

        // Create vertical spline once for this column
        s_vert.set_points(X_row, Y_row, tk::spline::cspline_hermite);

        for (int j = 0; j < 299 * 4 - 1; j++) {
            double y = j * 0.25;
            result[j][i] = s_vert(y);
        
        }
    }

    // Correct file output order
    for (int i = 0; i < 299 * 4 - 1; i++) {  
        for (int j = 0; j < 299 * 4 - 1; j++) {  
            splinec2 << result[i][j] << " ";
        }
        splinec2 << "\n";
    }

    splinec2.flush();
    splinec2.close();

    return 0;
}
