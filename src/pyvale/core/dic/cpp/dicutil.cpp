// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>

// Program Header files
#include "./dicutil.hpp"


namespace util {



    // displacements that can be accesses from anywhere
    double u;
    double v;
    double mag;


    void extract_image(std::vector<double> &image_def,
                       int *image_def_stack, 
                       int image_number,
                       int px_horizontal,
                       int px_vertical){

        // extract a single image
        std::vector<double> img(px_horizontal * px_vertical);
        int count = 0;


        for (int px_vert = 0; px_vert < px_vertical; px_vert++){
            for (int px_hori = 0; px_hori < px_horizontal; px_hori++){

                int index = image_number * px_horizontal * px_vertical + px_vert * px_horizontal + px_hori;
                image_def[count] = image_def_stack[index];
                // std::cout << image_def[count] << " ";
                count++;
            }
            // std::cout << std::endl;
        }
    }

    void extract_ss(std::vector<double> &image_def, 
                        std::vector<double> &ss, 
                        std::vector<double> &ss_coords_x,
                        std::vector<double> &ss_coords_y, 
                        int ss_x, 
                        int ss_y, 
                        int ss_size, 
                        int px_horizontal, 
                        int px_vertical){

        int count = 0;
        int index;
        for (int px_y = ss_y; px_y < ss_y+ss_size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_size; px_x++){

                // get coordinate values
                ss_coords_x[count] = px_x; 
                ss_coords_y[count] = px_y; 

                // get pixel values
                index = px_y * px_horizontal + px_x;
                ss[count] = image_def[index];
                // std::cout << ss_size << " " << ss_x << " " << ss_y << " " << " " << ss_coords_x[count] << " " << ss_coords_y[count] << " " << ss[count] << std::endl;
                count++;
                
            }
        }
    }


    std::vector<int> generate_ss_coord_list(bool *image_roi, int px_horizontal, int px_vertical, int ss_size, int ss_step){

        std::vector<int> ss_coord_list;
        int ss_x_min, ss_x_max, ss_y_min, ss_y_max;
        int index;

        for (int ss_y = 0; ss_y < px_vertical; ss_y+=ss_step){
            for (int ss_x = 0; ss_x < px_horizontal; ss_x+=ss_step){

                ss_x_min = ss_x - ss_size / 2;
                ss_x_max = ss_x + ss_size / 2;
                ss_y_min = ss_y - ss_size / 2;
                ss_y_max = ss_y + ss_size / 2;

                for (int px_y = ss_y_min; px_y <= ss_y_max; px_y++){
                    for (int px_x = ss_x_min; px_x <= ss_x_max; px_x++){

                        if (px_x < 0 || px_y < 0 || px_x >= px_horizontal || px_y >= px_vertical) 
                            goto next_ss;

                        index = px_y * px_horizontal + px_x;
                        if (image_roi[index] == false) 
                            goto next_ss;

                    }
                }
                ss_coord_list.push_back(ss_x);
                ss_coord_list.push_back(ss_y); 
                // std::cout << ss_x << " " << ss_y << std::endl;
                next_ss:;
            }
        }

        return ss_coord_list;
    }

    void resize_ss(std::vector<double> &ss, std::vector<double> &ss_x, std::vector<double> &ss_y, int ss_size) {
        ss.resize(ss_size * ss_size, 0.0);
        ss_x.resize(ss_size * ss_size, 0.0);
        ss_y.resize(ss_size * ss_size, 0.0);
    }

    


}   
