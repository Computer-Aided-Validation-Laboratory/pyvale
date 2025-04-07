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



    void extract_image(util::Image *image_def,
                       int *image_def_stack, 
                       int image_number){

        // extract a single image
        std::vector<double> img(image_def->px_horizontal * image_def->px_vertical);
        int count = 0;

        for (int px_vert = 0; px_vert < image_def->px_vertical; px_vert++){
            for (int px_hori = 0; px_hori < image_def->px_horizontal; px_hori++){

                int index = image_number * image_def->px_horizontal * image_def->px_vertical + px_vert * image_def->px_horizontal + px_hori;
                image_def->vals[count] = image_def_stack[index];
                // std::cout << image_def[count] << " ";
                count++;
            }
            // std::cout << std::endl;
        }
    }




    void extract_ss(int ss_x, int ss_y, util::Image *image_def, util::Subset *ss_def){

        int count = 0;
        int index;

        for (int px_y = ss_y; px_y < ss_y+ss_def->size; px_y++){
            for (int px_x = ss_x; px_x < ss_x+ss_def->size; px_x++){

                // get coordinate values
                ss_def->x[count] = px_x; 
                ss_def->y[count] = px_y; 

                // get pixel values
                index = px_y * image_def->px_horizontal + px_x;
                ss_def->vals[count] = image_def->vals[index];
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

}   
