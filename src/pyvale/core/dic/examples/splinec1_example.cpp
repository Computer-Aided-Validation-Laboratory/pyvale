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
    std::ofstream splinec1;
    std::stringstream sstr;
    sstr << "splinec1_output.dat";
    splinec1.open(sstr.str());


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


    // -------------------------------------------------------------------------------------------
    // Perform interpolation at a non-grid point
    // -------------------------------------------------------------------------------------------
    double x,y, interpolated_value;
    for (int i = 0; i < 299*4-1; i++){
        for (int j = 0; j < 299*4-1; j++){
            
            x = i * 0.25;
            y = j * 0.25;
            interpolated_value = splinec1::bicubicInterpolateCatmullRom(grid, px_vertical, px_horizontal, y, x);
            splinec1 << interpolated_value << " ";
        }
        splinec1 << "\n";
    }

    splinec1 << std::flush;
    splinec1.close();

    return 0;
}
