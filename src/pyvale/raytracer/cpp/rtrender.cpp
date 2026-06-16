// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <fstream>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>

// raytracer header files
#include "rtrender.h"
#include "rthitrecord.h"
#include "rtrayintersection.h"
#include "rtmaterials.h"

static constexpr double OFFSET_MAG = 1e7; // Secondary ray offset magnitude used to enlarge the base (machine epsilon, sitting at around 1e-16). To do: find the best value for that; current gets us to 1e-9

// ================================================================================
// Main loop for shooting rays and determining their colour
// ================================================================================

// Radiance with refractive materials - but we could make this into a separate option if refractive materials are present in the scene to avoid needing to branch into true/false hits if not necessary?
// This case would also have its own separate HitRecord, RayState structs since we could carry less data and fit more of those into cache lines
namespace renderer{ 

    // Set parameters with defaults
    int MAX_DEPTH = 2;
    EiVector3d background_color = EiVector3d::Zero();

    EiVector3d return_ray_color_stack(const Ray& primary_ray,
        const double scene_ri,
        const TLAS& TLAS){

        // Assign material interaction function pointer
        void (*ray_material_interaction_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color, const double offset); 

        //std::vector<RayState> stack; // Not thread safe
        thread_local std::vector<RayState> stack; // Thread safe
        stack.clear(); // Ensure clean state before starting
        stack.reserve(MAX_DEPTH);
        stack.emplace_back(primary_ray, scene_ri);
        EiVector3d total_color = EiVector3d::Zero(); // Starting ray colour

        // Iterate over rays until stack is empty or we terminate early (Russian roulette/hit MAX_DEPTH limit)
        while(!stack.empty()){
            RayState current_state = stack.back();
            stack.pop_back();
            const Ray& current_ray = current_state.ray;

            // Look for the intersection for this ray
            HitRecord intersection_record;
            const bool hit_anything = intersect_TLAS(current_ray, TLAS, intersection_record);

            // Determine volumetric absorption for the current ray segment
            EiVector3d absorption = EiVector3d::Zero(); // Set default absorption to 0.0 = clear medium
            // // Check if our ray has traversed any media that could attenuate the accumulated colour)
            if (current_state.interior_count > 0){ 
                absorption = interior_top_absorption(&current_state.interior_list[0], current_state.interior_count);
            }
            const bool has_absorption = absorption.x() > 0.0 || absorption.y() > 0.0 || absorption.z() > 0.0; // Save ourselves having to compute exponentials if there is no absorption

            // Handle escaping rays
            if (!hit_anything) {
                if (has_absorption) {
                    // Apply a huge distance to account for light lost traveling out of the scene
                    apply_absorption(current_state.accumulated_color, absorption, 1e30);
                }
                total_color += current_state.accumulated_color.cwiseProduct(renderer::background_color); // Sky colour
                //total_color += current_state.accumulated_color.cwiseProduct(ray_blue_sky(current_ray)); // Sky colour
                continue; // Early termination - no bounces here anyway
            }

            // Apply Beer-Lambert absorption globally for the traversed path segment
            if (has_absorption){
                // Since we store t from the ray equation ray(t) = origin_vector + t * direction_vector
                // t = (intersection_record.point_intersection - current_ray.origin).norm() (this has been confirmed within the code, too)
                // => path_length = t
                apply_absorption(current_state.accumulated_color, absorption, intersection_record.t);
            }
        

            // Assign material interaction function pointer
            ray_material_interaction_ptr = intersection_record.ray_material_ptr;

            // Adaptive offset to avoid self-intersection (shadow acne)
            // Calculated here as we use it here (nested dielectrics) as well as in material interactions
            // std::max guards against 0/underflow near the world origin (and users CAN place objects at origin)
            const double adaptive_offset = std::numeric_limits<double>::epsilon() * OFFSET_MAG *
                std::max({1.0,
                std::fabs(intersection_record.point_intersection.x()),
                std::fabs(intersection_record.point_intersection.y()),
                std::fabs(intersection_record.point_intersection.z())});

            // False hit detection from Schmidt's nested dielectrics algorithm (if using function pointer approach)
            // Shells don't participate in nested dielectrics (we enter-exit in one go, so they don't toggle the interior list)
            if (ray_material_interaction_ptr == &ray_refractive<ObjectType::SOLID>) {
                int hit_idx = intersection_record.hit_blas_idx;
                int hit_priority = intersection_record.hit_blas_priority;
                double hit_ri = intersection_record.refractive_index;
                const EiVector3d hit_absorption = intersection_record.face_color; // Face color = absorption for refractive materials
                
                // Get and compare the max priority currently surrounding the ray (or min value of int, if the list is empty)
                int top_idx = interior_highest_priority_idx(&current_state.interior_list[0], current_state.interior_count);
                int top_priority = (top_idx < 0) ? std::numeric_limits<int>::min() : current_state.interior_list[top_idx].priority;
                
                // Check if it is a true hit
                if (current_state.interior_count > 0 && hit_priority < top_priority) {
                    // False hit: this boundary is inside a higher-priority medium
                    // Toggle the list (track that we crossed it) but do not shade
                    // Then re-cast the ray from hit point in the same direction by pushing a new RayState whose origin is the current hit point
                    //std::cerr << "Inside interior count check" << std::endl;
                    RayState next_state = current_state;
                    interior_toggle(&next_state.interior_list[0], next_state.interior_count, hit_idx, hit_priority, hit_ri, hit_absorption);
                    // Offset the ray minimnally to avoid self-intersecting - much like we do for all secondary rays
                    next_state.ray.origin = intersection_record.point_intersection + adaptive_offset * current_ray.direction;
                    next_state.ray.direction = current_ray.direction;
                    next_state.ray.t_min = SPAWNED_T_MIN_BASE * std::max(1.0, intersection_record.point_intersection.norm());
                    next_state.ray.t_max = std::numeric_limits<double>::infinity();
                    // DO NOT INCREMENT DEPTH - false hits are invisible according to the paper, so they do not affect the ray bounce count or energy
                    stack.push_back(next_state);
                    continue;
                }
            }

            const bool is_refractive = (intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SOLID>
                                    || intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SHELL>);
            
            
            // Explicit depth limit with ambient fallback
            if (current_state.depth >= MAX_DEPTH) {
                // Add a fallback ambient color to compensate for truncated energy 
                // Avoids the "plain black shadows" caused by zero light return
                EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2; 
                total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission + ambient_fallback);
                continue; 
            }
            
            EiVector3d albedo = intersection_record.face_color;
            if (current_state.depth > MAX_DEPTH/2) { // Start early termination if we are at least halfway through the maximum allowed depth
                // Russian roulette early termination
                // For refractive materials, face_color is an absorption coefficient (sigma_a),
                // not a reflectance - using it as a survival probability causes strongly-tinted dielectrics to be terminated often
                // We use albedo = (1,1,1) for refractive materials so RR never fires against them here;
                // ray_refractive handles its own internal RR separately.
            
                // Clamp to prevent infinite loops (p=1.0) and division by zero (p=0.0)
                EiVector3d rr_albedo = is_refractive ? EiVector3d(1.0, 1.0, 1.0) : albedo;
                double p = std::clamp(rr_albedo.maxCoeff(), 0.1, 0.95);
                if (random_double() > p){ 
                //if ((double)rand() / RAND_MAX > p){ // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
                    total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission);
                    continue;
                }
                // Only rescale albedo for non-refractive materials; dielectrics pass albedo=attenuation=(1,1,1)
                // into ray_refractive regardless, so scaling face_color here would corrupt sigma_a
                // Note: Currently ray_refractive knows that it shouldn't use albedo, so we do not need this check
                // However, if you were to directly overwrite intersection_record.face_color and remove albedo from ray_material_interaction_ptr,
                // then yes, this check would be necessary
                //if (!(is_refractive)){
                //    albedo /= p;
                //}
            }

            // True hit: priority of hit object >= max priority in interior list OR the list is empty
            // Shade normally - FUNCTION POINTER VARIANT
            ray_material_interaction_ptr(current_state, intersection_record, albedo, stack, total_color, adaptive_offset);

            // SWITCH DISPATCH VARIANT
            // Requires updating rtbvh.cpp and .h, rthitrecord, rtrayintersection (intersect_BLAS) to store material & object_type data
            // During profiling in May 2026, this showed marginally fewer L1 instruction cache misses (0-0.01 percent point difference in all test runs), BUT the average runtime was 1.58% worse
            // This was not enough to rule out this approach definitely, so while the function pointer approach was retained as the default, this is kept commented out in case
            /*
            switch (intersection_record.material) {
                case UNLIT: {
                ray_unlit(current_state, intersection_record, albedo, stack, total_color);
                    break;
                }
                case DIFFUSE: { // Diffuse
                    ray_diffuse(current_state, intersection_record, albedo, stack, total_color);
                    break;
                }
                case SPECULAR: {// Specular (mirror)
                    ray_specular(current_state, intersection_record, albedo, stack, total_color);
                    break;
                }
                case REFRACTIVE: {// Refraction (dielectric)
                    // Check for false hit
                    int hit_idx = intersection_record.hit_blas_idx;
                    int hit_priority = intersection_record.hit_blas_priority;
                    double hit_ri = intersection_record.refractive_index;
                    // Get and compare the max priority currently surrounding the ray (or min value of int, if the list is empty)
                    int top_idx = interior_highest_priority_idx(&current_state.interior_list[0], current_state.interior_count);
                    int top_priority;

                    if (top_idx < 0){
                        top_priority = std::numeric_limits<int>::min();
                    }
                    else {
                        top_priority = current_state.interior_list[top_idx].priority;
                    }
                    
                    // Check if it is a true hit
                    if (!(current_state.interior_count == 0 || hit_priority >= top_priority)){
                        // False hit: priority of hit object < max priority in interior list (Schmidt's algorithm for nested volumes)
                        // => Do not shade; re-cast the ray from hit point in the same direction by pushing a new RayState whose origin is the current hit point
                        RayState next_state = current_state;
                        interior_toggle(&next_state.interior_list[0], next_state.interior_count, hit_idx, hit_priority, hit_ri, intersection_record.face_color);
                        // Offset the ray minimnally to avoid self-intersecting - much like we do for all secondary rays
                        const double offset = std::numeric_limits<double>::epsilon() * 10.0 *
                            std::max({std::fabs(intersection_record.point_intersection.x()),
                            std::fabs(intersection_record.point_intersection.y()),
                            std::fabs(intersection_record.point_intersection.z())});
                        next_state.ray.origin = intersection_record.point_intersection + offset * current_ray.direction;
                        next_state.ray.direction = current_ray.direction;
                        next_state.ray.t_min = current_ray.t_min;
                        next_state.ray.t_max = std::numeric_limits<double>::infinity();
                        // DO NOT INCREMENT DEPTH - false hits are invisible according to the paper, so they do not affect the ray bounce count or energy
                        stack.push_back(next_state);
                        continue;
                    }
                    if (intersection_record.object_type == ObjectType::SOLID){
                        ray_refractive<ObjectType::SOLID>(current_state, intersection_record, albedo, stack, total_color);
                    }
                    else{
                        ray_refractive<ObjectType::SHELL>(current_state, intersection_record, albedo, stack, total_color);
                    }
                    break;
                }
            }*/
            
        } // Stack while loop
        return total_color;
    } 

    void set_depth(const int max_depth){
        MAX_DEPTH = max_depth;
    }

    void set_background(const EiVector3d& color){
        background_color = color;
    }
}

// ================================================================================
// Writing the output
// ================================================================================
namespace outputwriter{

    void (*save_image)(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath);

    // Helper that writes 16-bit integers in Little-Endian byte order
    static inline void write_16bit(std::ofstream& image_file,
        uint16_t value){

        uint8_t bytes[2] = {static_cast<uint8_t>(value & 0xFF), static_cast<uint8_t>(value >> 8) };
        image_file.write(reinterpret_cast<const char*>(bytes), 2);
    }

    // Helper that writes 32-bit integers in Little-Endian byte order
    static inline void write_32bit(std::ofstream& image_file,
        uint32_t value){

        uint8_t bytes[4] = { static_cast<uint8_t>(value & 0xFF),
        static_cast<uint8_t>((value >> 8) & 0xFF),
        static_cast<uint8_t>((value >> 16) & 0xFF),
        static_cast<uint8_t>((value >> 24) & 0xFF) };
        image_file.write(reinterpret_cast<const char*>(bytes), 4);
    }

    // Helper that writes a TIFF tag (12-byte IFD tag)
    static inline void write_tag(std::ofstream& image_file,
        uint16_t tag,
        uint16_t type,
        uint32_t count,
        uint32_t value) {

        write_16bit(image_file, tag);
        write_16bit(image_file, type);
        write_32bit(image_file, count);
        write_32bit(image_file, value);
    }

    void saveBMP_24bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {
            
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".bmp");

        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // BMP rows must be a multiple of 4 bytes. Calculate necessary padding
        int row_bytes = image_width * 3; // 24-bit mode => 3 bytes/pixel
        int padding = (4 - (row_bytes % 4)) % 4;
        uint32_t pixel_data_size = (row_bytes + padding) * image_height;
        uint32_t file_size = 54 + pixel_data_size; // 14 (file header) + 40 (info header) + pixels

        // Write the BMP file header (14 bytes)
        image_file.write("BM", 2); // Signature
        write_32bit(image_file, file_size); // File Size
        write_32bit(image_file, 0); // Reserved
        write_32bit(image_file, 54); // Offset to pixel data (14 + 40)

        // Write the DIB Info Header (40 bytes)
        write_32bit(image_file, 40); // Info Header size
        write_32bit(image_file, image_width); // Image width
        write_32bit(image_file, image_height); // Image height (positive = bottom-up)
        write_16bit(image_file, 1); // Color planes (must be 1)
        write_16bit(image_file, 24); // Bits per pixel (24 for RGB)
        write_32bit(image_file, 0); // Compression (0 = None / BI_RGB)
        write_32bit(image_file, pixel_data_size); // Image size (including padding)
        write_32bit(image_file, 2835); // X pixels per meter (approx 72 DPI)
        write_32bit(image_file, 2835); // Y Pixels per meter (approx 72 DPI)
        write_32bit(image_file, 0); // Total colors (0 = default)
        write_32bit(image_file, 0); // Important colors (0 = default)

        // Write pixel data
        uint8_t pad_bytes[3] = {0, 0, 0};

        // Iterate backwards through rows for bottom-up writing
        for (int y = image_height - 1; y >= 0; --y) {
            for (int x = 0; x < image_width; ++x) {
                int i = (y * image_width + x) * 3;
                
                // Extract RGB from buffer and write as BGR
                char b = static_cast<char>(pixel_buffer[i + 2]);
                char g = static_cast<char>(pixel_buffer[i + 1]);
                char r = static_cast<char>(pixel_buffer[i]);

                image_file.put(b);
                image_file.put(g);
                image_file.put(r);
            }
            // Add required 4-byte alignment padding at the end of the row
            if (padding > 0) {
                image_file.write(reinterpret_cast<const char*>(pad_bytes), padding);
            }
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    void saveBMP_8bit(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath) {
            
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".bmp");

        std::ofstream image_file(output_filepath, std::ios::binary);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        // 8-bit BMP rows must be a multiple of 4 bytes. Calculate padding.
        int row_bytes = image_width; // 1 byte per pixel in 8-bit mode
        int padding = (4 - (row_bytes % 4)) % 4;
        uint32_t pixel_data_size = (row_bytes + padding) * image_height;
        
        // File sizes
        uint32_t header_size = 14;
        uint32_t info_header_size = 40;
        uint32_t palette_size = 256 * 4; // 256 colors * 4 bytes (B, G, R, Reserved)
        uint32_t offset_to_pixels = header_size + info_header_size + palette_size;
        uint32_t file_size = offset_to_pixels + pixel_data_size; 

        // Write the BMP file header (14 bytes)
        image_file.write("BM", 2); // Signature
        write_32bit(image_file, file_size); // File size
        write_32bit(image_file, 0); // Reserved
        write_32bit(image_file, offset_to_pixels); // Offset to pixel data (1078)

        // Write the DIB info header (40 bytes)
        write_32bit(image_file, info_header_size); // Info header size
        write_32bit(image_file, image_width); // Image width
        write_32bit(image_file, image_height); // Image jeight (Positive = bottom-up)
        write_16bit(image_file, 1); // Color planes (must be 1)
        write_16bit(image_file, 8); // Bits per pixel (8 for indexed)
        write_32bit(image_file, 0); // Compression (0 = None)
        write_32bit(image_file, pixel_data_size); // Image size (including padding)
        write_32bit(image_file, 2835); // X pixels per meter (approx 72 DPI)
        write_32bit(image_file, 2835); // Y pixels per meter (approx 72 DPI)
        write_32bit(image_file, 256); // Total colors in palette (256)
        write_32bit(image_file, 256); // Important colors (256)

        // Write the 256-colour grayscale palette (1024 bytes)
        // This is needed so the viewer can map 8-bit index to a screen colour
        // Palette format: Blue, Green, Red, Reserved (0)
        for (int i = 0; i < 256; ++i) {
            char gray_val = static_cast<char>(i);
            image_file.put(gray_val); // Blue
            image_file.put(gray_val); // Green
            image_file.put(gray_val); // Red
            image_file.put(0); // Reserved byte
        }

        // Write pixel data
        uint8_t pad_bytes[3] = {0, 0, 0};

        // Iterate backwards through rows for bottom-up writing
        for (int y = image_height - 1; y >= 0; --y) {
            for (int x = 0; x < image_width; ++x) {
                // Buffer is formatted as 3 bytes per pixel (R, G, B)
                int i = (y * image_width + x) * 3;
                
                // In grayscale mode, R=G=B, so we just grab the first byte (Red channel)
                // This acts as the 8-bit index into our grayscale palette
                char pixel_index = static_cast<char>(pixel_buffer[i]);
                image_file.put(pixel_index);
            }
            // Add required 4-byte alignment padding at the end of the row
            if (padding > 0) {
                image_file.write(reinterpret_cast<const char*>(pad_bytes), padding);
            }
        }

        image_file.close();
        std::cout << "\r Done. \n";
    }

    void savePPM(const std::vector<uint8_t>& pixel_buffer,
        const int image_height,
        const int image_width,
        std::filesystem::path& output_filepath){
        
        // Finish the output filepath with the appropriate extension
        output_filepath.concat(".ppm");
        //std::cout << "Output filepath:" << output_filepath << std::endl; // For checking if path is generated correctly

        std::ofstream image_file;

        image_file.open(output_filepath);
        if (!image_file.is_open()) {
            std::cerr << "Failed to open the output file.\n";
            return;
        }

        image_file << "P6\n" << image_width << ' ' << image_height << "\n255\n";
        image_file.write(reinterpret_cast<const char*>(pixel_buffer.data()), pixel_buffer.size());

        image_file.close();
        std::cout << "\r Done. \n";
    }

    // Setter
    void set(OutputFormat output_format, ChannelCount channel_count){
        switch (output_format){
            case OutputFormat::PPM:
                save_image = &savePPM;
                break;
            case OutputFormat::TIFF_8BIT:
                switch(channel_count){
                    case(ChannelCount::MONO):
                        save_image = &saveTIFF_8bit<ChannelCount::MONO>; break;
                    case (ChannelCount::RGB):
                        save_image = &saveTIFF_8bit<ChannelCount::RGB>; break;
                }
                break;
            //case OutputFormat::TIFF_16BIT:
            //    switch(channel_count){
            //        case(ChannelCount::MONO):
            //            save_image = &saveTIFF_16bit<ChannelCount::MONO>; break;
            //        case (ChannelCount::RGB):
            //            save_image = &saveTIFF_16bit<ChannelCount::RGB>; break;
            //    }
            //    break;
            case OutputFormat::BMP_24BIT:
                save_image = &saveBMP_24bit;
                break;
            case OutputFormat::BMP_8BIT:
                save_image = &saveBMP_8bit;
                break;
            //case OutputFormat::NP_BUFFER:
                //save_image = &saveNPBuffer;
                //break;
            default:
                save_image = &saveBMP_8bit;
                break;
        }
    }

}

// ================================================================================
// Debug helper
// ================================================================================

void mock_ray_shoot(const EiVector3d& camera_center,
    const EiVector3d& pixel_00_center,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_pixel_spacing,
    const Eigen::Matrix<double, 2, 3, Eigen::StorageOptions::RowMajor>& matrix_defocus_disc,
    const TLAS& TLAS,
    const int image_height,
    const int image_width,
    const int number_of_samples,
    const double scene_ri,
    const std::filesystem::path output_filepath) {
    // Shoot a single mock ray to see what happens - helpful in debugging

    Ray mock_ray;
    //mock_ray.origin = EiVector3d(0.0, 0.0, 410.0); wedge tests
    //mock_ray.direction = EiVector3d(0.15963, -0.0311445, -0.986686);
    mock_ray.origin = EiVector3d(0.0, 0.0, 410.0); //normals tests
    mock_ray.direction = EiVector3d(0.0814686, 0.171913, -0.981738);
    EiVector3d pixel_color = EiVector3d::Zero();
    pixel_color += renderer::return_ray_color_stack(mock_ray, scene_ri, TLAS);
    std::cerr << "Final color: " << pixel_color.x() << ", " << pixel_color.y() << ", " << pixel_color.z() << std::endl;
}

// ================================================================================
// Old versions of return_ray_color in case they are helpful for debugging/dev
// ================================================================================

// Same as above but withous nested dielectrics, i.e., pure Beer-Lambert
/*
EiVector3d return_ray_color_stack(const Ray& primary_ray,
    const double scene_ri,
    const TLAS& TLAS){

    EiVector3d total_color = EiVector3d::Zero();
    thread_local std::vector<RayState> stack;
    stack.clear();
    stack.reserve(MAX_DEPTH);
    stack.emplace_back(primary_ray, scene_ri);

    void (*ray_material_interaction_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color, const double offset);

    while(!stack.empty()){
        RayState current_state = stack.back();
        stack.pop_back();
        const Ray& current_ray = current_state.ray;

        HitRecord intersection_record;
        IntersectionOutput intersection;
        const bool hit_anything = intersect_TLAS(current_ray, TLAS, intersection, intersection_record);

        if (!hit_anything) {
            total_color += current_state.accumulated_color.cwiseProduct(ray_blue_sky(current_ray));
            continue;
        }

        // Determine if the intersected material is refractive.
        // Refractive materials manage their own absorption (Beer-Lambert) internally,
        // so we must NOT apply it here — doing so causes double-absorption and black output.
        const bool is_refractive = (intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SOLID>
                                 || intersection_record.ray_material_ptr == &ray_refractive<ObjectType::SHELL>);


        // Assign material interaction function pointer
        ray_material_interaction_ptr = intersection_record.ray_material_ptr;

        // Adaptive offset to avoid self-intersection (shadow acne).
        // std::max guards against underflow near the world origin.
        const double adaptive_offset = std::numeric_limits<double>::epsilon() * OFFSET_MAG *
            std::max({1.0,
            std::fabs(intersection_record.point_intersection.x()),
            std::fabs(intersection_record.point_intersection.y()),
            std::fabs(intersection_record.point_intersection.z())});

        // Hard depth cap with ambient fallback to avoid pure-black truncated paths
        if (current_state.depth >= MAX_DEPTH) {
            EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2;
            total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission + ambient_fallback);
            continue;
        }

        EiVector3d albedo = intersection_record.face_color;

        if (current_state.depth > MAX_DEPTH / 2) {
            // Russian roulette early termination
            // For refractive materials, face_color is an absorption coefficient (sigma_a), not a reflectance
            // We use albedo = attenuation = (1,1,1) for refractive materials so Russian roulette (RR) never fires against them here;
            // ray_refractive handles its own internal RR separately.
            EiVector3d rr_albedo = is_refractive ? EiVector3d(1.0, 1.0, 1.0) : albedo;
            double p = std::clamp(rr_albedo.maxCoeff(), 0.1, 0.95);
            if (random_double() > p) {
                total_color += current_state.accumulated_color.cwiseProduct(intersection_record.emission);
                continue;
            }
            // Only rescale albedo for non-refractive materials; dielectrics pass attenuation=(1,1,1)
            // into ray_refractive regardless, so scaling face_color here would corrupt the absorption coefficient stored in the same variable
            if (!is_refractive) {
                albedo /= p;
            }
        }

        ray_material_interaction_ptr(current_state, intersection_record, albedo, stack, total_color, adaptive_offset);

    } // Stack while loop

    return total_color;
}
*/

/*
// Previous version without nested refractive materials or Beer-Lambert
EiVector3d return_ray_color_stack_nr(const Ray& primary_ray, const double scene_ri, const TLAS& TLAS){
    EiVector3d total_color = EiVector3d::Zero();
    std::vector<RayState> stack;
    stack.reserve(MAX_DEPTH);
    stack.emplace_back(primary_ray, scene_ri);
    void (*ray_material_interaction_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color); // Pointer to the function determining the interaction between the ray and the mesh material

    while(!stack.empty()){
        RayState current_state = stack.back();
        stack.pop_back();
        current_state.ray.direction.stableNormalize();
        Ray current_ray = current_state.ray;

        HitRecord intersection_record; // Create HitRecord struct
        // Look for the first intersection for this ray
        IntersectionOutput intersection;
        intersect_TLAS(current_ray, TLAS, intersection, intersection_record);

        ray_material_interaction_ptr = intersection_record.ray_material_ptr;
        //intersection_record.temp_flat_shading(); // Temporary function to swap shading normal with geometric normal and test flat shading before it is implemented as its own separate option

        if (intersection_record.t == std::numeric_limits<double>::infinity()) {
            //const EiVector3d blue_sky(0.5, 0.5, 0.5);
            const EiVector3d blue_sky = ray_blue_sky(current_ray); // Early termination - no bounces here anyway
            total_color += current_state.accumulated_color.cwiseProduct(blue_sky);
            continue;
        }
        
        EiVector3d emitted = intersection_record.emission;
        EiVector3d albedo = intersection_record.face_color;
        
        // Explicit depth limit with ambient fallback
        if (current_state.depth >= MAX_DEPTH) {
        //    // Add a fallback ambient color to compensate for truncated energy 
            // Avoids the "plain black shadows" caused by zero light return
            EiVector3d ambient_fallback = ray_blue_sky(current_ray) * 0.2; 
            total_color += current_state.accumulated_color.cwiseProduct(emitted + ambient_fallback);
            continue; 
        }
        
        if (current_state.depth > MAX_DEPTH/2) { // Start early termination if we are at least halfway through the maximum allowed depth
            // Russian roulette early termination
            double p = std::max({albedo.x(), albedo.y(), albedo.z()});
            // Clamp to prevent infinite loops (p=1.0) and division by zero (p=0.0)
            p = std::clamp(p, 0.05, 0.95); 
            if (random_double() > p){  // Note: for multi-threading this will have to be replaced with thread_local generator
            //if ((double)rand() / RAND_MAX > p){ // std rand() won't work if we multi-thread this (mutex lock) + has poor statistical distribution
                //return emitted;
                total_color += current_state.accumulated_color.cwiseProduct(emitted);
                continue;
            }
            albedo /= p;
        }
        // Process ray and update the stack and total color based on the material of the intersected mesh
        ray_material_interaction_ptr(current_state, intersection_record, albedo, stack, total_color);
    } // Stack while loop
    return total_color;
} 
*/

/*
// This a new radiance function with lighting
EiVector3d return_ray_color_new(const Ray& ray,
                           const TLAS& TLAS,
                           int depth = 0) {

    HitRecord rec; // Create HitRecord struct
    // Look for intersection
    IntersectionOutput intersection;
    intersect_TLAS(ray, TLAS, intersection, rec);

    // rec.material = DIFFUSE;

    // Blue sky gradient
    if (rec.t == std::numeric_limits<double>::infinity() 
        || rec.material == NOT_DEFINED) {
        double a = 0.5 * (ray.direction(1) + 1.0);
        static EiVector3d white, blue;
        white << 1.0, 1.0, 1.0;
        blue << 0.5, 0.7, 1.0;
        return (1.0 - a) * white + a * blue;
    }

    set_face_normal(ray, rec.normal_surface);

    EiVector3d emitted = rec.emission;
    EiVector3d albedo = rec.face_color;

    // Russian roulette
    double p = std::max({albedo.x(), albedo.y(), albedo.z()});
    if (depth > 5) {
        if ((double)rand() / RAND_MAX > p)
            return emitted;
        albedo /= p;
    }

    switch (rec.material) {

    case UNLIT: {
        return rec.face_color;

    }

    case DIFFUSE: { // Diffuse
        EiVector3d w = rec.normal_surface;

        EiVector3d u =
            ((fabs(w.x()) > 0.1 ? EiVector3d(0,1,0) : EiVector3d(1,0,0))
            .cross(w)).normalized();

        EiVector3d v = w.cross(u);

        double r1 = 2 * M_PI * ((double)rand() / RAND_MAX);
        double r2 = (double)rand() / RAND_MAX;
        double r2s = sqrt(r2);

        EiVector3d d =
            (u * cos(r1) * r2s +
             v * sin(r1) * r2s +
             w * sqrt(1 - r2)).normalized();

        Ray ray_new;
        ray_new.origin = rec.point_intersection;
        ray_new.direction = d;

        return emitted + albedo.cwiseProduct(
            return_ray_color_new(ray_new, TLAS, depth + 1)
        );
    }

    case SPECULAR: {// Specular (mirror)
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(rec.normal_surface) * rec.normal_surface;
        
        Ray ray_new;
        ray_new.origin = rec.point_intersection;
        ray_new.direction = reflected;

        return emitted + albedo.cwiseProduct(
            return_ray_color_new(ray_new, TLAS, depth + 1)
        );
    }

    case REFRACTIVE: {// Refraction (dielectric)
        EiVector3d n = rec.normal_surface;
        EiVector3d nl = (ray.direction.dot(n) < 0) ? n : -n;
    
        EiVector3d reflected =
            ray.direction - 2 * ray.direction.dot(n) * n;
    
        Ray reflected_ray;
        reflected_ray.origin = rec.point_intersection;
        reflected_ray.direction = reflected;
    
        bool into = ray.direction.dot(nl) < 0; // entering or exiting
    
        double ri_surrounding = 1.0;   // air
        double ri_material = 1.5;   // glass
        double ri_ratio = into ? ri_surrounding / ri_material : ri_material / ri_surrounding;
    
        double dot_incidence = ray.direction.dot(nl);
        double cos_transmission2 = 1 - ri_ratio * ri_ratio * (1 - dot_incidence * dot_incidence);
    
        // Total internal reflection
        if (cos_transmission2 < 0) {
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflected_ray, TLAS, depth + 1)
            );
        }
    
        EiVector3d dir_transmission =
            (ray.direction * ri_ratio -
             n * ((into ? 1 : -1) * (dot_incidence * ri_ratio + sqrt(cos_transmission2)))).normalized();
    
        // Schlick approximation
        double a = ri_material - ri_surrounding;
        double b = ri_material + ri_surrounding;
        double R0 = (a * a) / (b * b);
    
        double c = 1 - (into ? -dot_incidence : dir_transmission.dot(n));
        double reflectance = R0 + (1 - R0) * c * c * c * c * c;
        double transmittance = 1 - reflectance;
    
        // Russian roulette between reflection and refraction
        double P = 0.25 + 0.5 * reflectance;
        double P_reflect = reflectance / P;
        double P_transmit = transmittance / (1 - P);
    
        if (depth > 2) {
            if ((double)rand() / RAND_MAX < P) {
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(reflected_ray, TLAS, depth + 1) * P_reflect
                );
            } else {
                Ray refracted_ray;
                refracted_ray.origin = rec.point_intersection;
                refracted_ray.direction = dir_transmission;
    
                return emitted + albedo.cwiseProduct(
                    return_ray_color_new(refracted_ray, TLAS, depth + 1) * P_transmit
                );
            }
        } else {
            Ray refracted_ray;
            refracted_ray.origin = rec.point_intersection;
            refracted_ray.direction = dir_transmission;
    
            return emitted + albedo.cwiseProduct(
                return_ray_color_new(reflected_ray, TLAS, depth + 1) * reflectance +
                return_ray_color_new(refracted_ray, TLAS, depth + 1) * transmittance
            );
        }
    }
    }

    return emitted;
}
*/
