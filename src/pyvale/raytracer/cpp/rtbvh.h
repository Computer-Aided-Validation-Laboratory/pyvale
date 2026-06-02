// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#pragma once
// STD header files
#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <string>
#include <iostream>

// nanobind header files
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

// raytracer header files
#include "rteigentypes.h"
#include "rtray.h"
#include "rtelemconstants.h"
#include "rthitrecord.h"
#include "rtmaterials.h"


// Bounding volume structure - axis-aligned bounding boxes (AABB)
// Struct size: 2 x 8 x 3 = 48 bytes
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }

    // Used for building AABBs for all mesh elements, regardless of the type
    inline void build_for_element(const double* element_node_coords, const int element_node_count){
        // Iterate through each element node
        for (int node = 0; node < element_node_count; ++node){
            const int offset = node * NODE_COORDINATES;
            // Iterate through all coordinates (x, y, z)
            for (int i = 0; i < 3; ++i) {
                const double nodal_coordinate = element_node_coords[offset + i];
                corner_min[i] = std::min(corner_min[i], nodal_coordinate);
                corner_max[i] = std::max(corner_max[i], nodal_coordinate);
            }
        }
    }

     // Used for SAH splitting
    inline void expand_to_include_point(const std::array<double,3>& point){
        // Iterate over x, y, z coordinates
        for (int i = 0; i < NODE_COORDINATES; ++i){
            double point_coordinate = point[i];
            corner_min[i] = std::min(corner_min[i], point_coordinate);
            corner_max[i] = std::max(corner_max[i], point_coordinate);
        }
    }
     // Used for creating child node AABBs
    inline void expand_to_include_AABB(const AABB& other) {
        // Iterate over x, y, z coordinates
        for (int i = 0; i < NODE_COORDINATES; ++i){
            corner_min[i] = std::min(corner_min[i], other.corner_min[i]);
            corner_max[i] = std::max(corner_max[i], other.corner_max[i]);
        }
    }

    inline double find_axis_extent(int axis) const {
        double result = corner_max[axis] - corner_min[axis];
        if (result < 0) return 0.0;
        return result;
    }
    inline double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        // Surface area of rectangular prism
        return 2 * (height * width + width * depth + height * depth);
    }

    inline double find_diagonal() const{
        // Diagonal of rectangular prism
        return std::sqrt(std::pow(corner_max[0] - corner_min[0], 2) + std::pow(corner_max[1] - corner_min[1], 2) + std::pow(corner_max[2] - corner_min[2], 2));
    }
};

// Struct size: 48 + 4 = 52 bytes
struct Bin {
    // Bin for binning SAH
    AABB bounding_box {};
    int element_count {0};
};

// Struct used as a temporary data carrier in build_BLAS and build_TLAS
// Struct size: 8 (64-bit system) + 4 x 2 = 16 bytes
struct BuildTask {
    size_t element_count; // Number of elements
    int node_idx;
    int min_element_idx; // First triangle index in element_indices

    BuildTask() = default;
    BuildTask(size_t element_count, int node_idx, int min_element_idx):
        element_count(element_count),
        node_idx(node_idx),
        min_element_idx(min_element_idx)
        {};
};

// Struct size: 8 (64-bit system) + 4 x 2 = 16 bytes
struct Texture {
    const double* data {nullptr}; // Pointer to the texture, so we can just assign it to relevant BVH nodes and sample
    int height {0};
    int width {0};
    
    // Default constructor
   Texture() = default;
   Texture(const double* data, int height, int width):
    data(data),
    height(height),
    width(width)
    {};
};

// BLAS - Bottom Level Acceleration Structure. Each BLAS stores a BVH for one mesh in the scene
// Struct size in worst case: 
// Each vector: MAX_ELEMENTS_PER_LEAF (currently = 4) x  nodes_per_element (max. 9 for QUAD9) x 3 (x,y,z per node) x 8 (double) = 864 bytes
// Rest: 48 + 8 (64-bit system) + 4 + 1
// Total: 864 x 3 + 48 + 8 + 4 + 1 = 2653 bytes = 2.653 kB
struct BLAS_Node {
    std::vector<double> node_coords; // Coordinates of nodes comprising the mesh elements stored in the node, if applicable
    std::vector<double> node_normals; // Normals of nodes comprising the mesh elements stored in the node, if applicable
    std::vector<double> face_color; // Either (faces, 3) array with color values or (faces,2) array with (u,v) coordinates
    AABB bounding_box {};
    size_t element_count {0}; // If not zero, this is the leaf
    int left_child_idx {-1};
    enum ElementNodeCount nodes_per_element {ElementNodeCount::TRI3}; // Default to triangles

    // Constructors for emplace_back to avoid temporary copies
   BLAS_Node() = default;
   BLAS_Node(AABB aabb, size_t element_count, int left_child_idx):
    bounding_box(aabb),
    element_count(element_count),
    left_child_idx(left_child_idx)
    {}; 
};

// Forward declaration (incomplete types) so we can use them in function pointers in BLAS while avoiding circular dependencies (since they depend on BLAS_Node defined here)
struct IntersectionOutput; 

// Struct size (worst case):
// Vector: number of elements in the mesh * 2653 bytes
// Rest: 48 + 16 + 8 (64-bit system) + 8 x 2 + 8 (64-bit system) x 2 + 4 x 3 = 116 bytes
struct BLAS {
    std::vector<BLAS_Node> tree_nodes;
    AABB bounding_box {};
    // We cannot just copy relevant pieces of texture into different BLAS nodes, so keep it at the BLAS level. This will also allow us to do fewer if/else checks in intersection for coloring
    Texture texture {}; // If texture.data is not a nullptr, face_color is (u,v). This logic saves us having to store surface type explicitly
    IntersectionOutput (*intersection_function_ptr)(const Ray&, const std::vector<double>& node_coords, const unsigned int bvh_node_element_count) {nullptr}; // Ray-mesh element intersection (TRI3, QUAD4, etc.)
    double refractive_index {1.0}; // Refractive index of the mesh material; set to 1.0 to avoid bad division in case it somehow gets unitialised
    double thickness {1.0}; // Thickness of a SHELL mesh; unused for solids
    // Void function pointer will be 8 bytes in 64-bit system, 4 in a 32x, so this should be the best positioning of those for memory alignment
    void (*overwrite_intersection_function_ptr)(HitRecord&, const BLAS_Node&, const Texture& texture, Eigen::Index min_row_idx) {nullptr}; // Saving data to HitRecord depending on the surface type (color/texture) and element type
    void (*ray_material_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color) {nullptr}; // Pointer to the function determining the interaction between the ray and the mesh material
    // Uncomment the below 2 lines if deciding to go for switch-based dispatch in return_ray_color
    //ObjectType object_type;
    //int material;
    int priority; // Priority - tells us the ordering of nested volumes in the scene
    int root_idx {-1};
    int blas_idx {-1}; // ID in TLAS, used for handling nested refractive volumes without OOP/pointers

    BLAS() = default; // Constructor for emplace_back to avoid temporary copies
};

// TLAS - Top Level Acceleration Structure. Stores all BLASes for the scene, used for preliminary intersection
// Struct size: 48 + 4 x 3 = 60 bytes
struct TLAS_Node {
    AABB bounding_box {};
    int blas_count {0}; // Number of BLASes in this node (consecutive in the array)
    int left_child_idx {-1};
    int min_blas_idx {-1}; // Store this instead of data as we expect a few meshes in the scene tops, so indexing into BLAS vector shouldn't be too awful

     // Constructors for emplace_back to avoid temporary copies
    TLAS_Node() = default;
    TLAS_Node(AABB aabb, int count, int left_idx, int min_blas_idx):
        bounding_box(aabb),
        blas_count(count),
        left_child_idx(left_idx),
        min_blas_idx(min_blas_idx)
        {};
};

struct TLAS {
    std::vector<BLAS> blases;
    std::vector<TLAS_Node> tlas_nodes;
};

inline void compute_element_centroid(const double *element_node_coords, // Pointer to an array, so we can have one centroid function to rule them all without having to specify the array size here
    std::array<double, NODE_COORDINATES> &element_centroid,
    int element_node_count);

inline void compute_mesh_centroid(AABB& mesh_aabb, std::array<double,3>& mesh_centroid);

template<ElementNodeCount element_node_count>
void process_element_data(size_t mesh_number_of_elements,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_element_aabbs,
    AABB& mesh_aabb,
    const int timestep){

    // We need these to be compile time constants to create arrays. They are known from switch in build_acceleration_structures
    constexpr int nodes_per_element = static_cast<int>(element_node_count); // Explicitly cast to int
    constexpr int coords_per_element = nodes_per_element * NODE_COORDINATES; // number of elements times 3 coordinates (x,y,z) each. size_t only to be able to pass this as an argument in array creation
    const int timestep_stride = timestep * mesh_number_of_elements * nodes_per_element * NODE_COORDINATES;

    // Allocate arrays once, then simpply keep overwriting data for each element
    std::array<double, coords_per_element> element_node_coords;
    std::array<double, NODE_COORDINATES> element_centroid;

     // Iterate over elements comprising a mesh
    for (int element_idx = 0; element_idx < mesh_number_of_elements; element_idx++) {
        // Use pointers - means we treat the 2D array as a flat 1D array and do the indexing manually by calculating the offset.
        // HAS to be contiguous in memory for this to work properly! c_contig flag in nanobind ensures that
        int element_min_index = timestep_stride + element_idx * coords_per_element; // Find the minimum index corresponding to the given element at given timestep
        
        // Gather coordinates for all nodes in the considered element
        // element_node_coords is structured as [x0, y0, z0, x1, y1, z1, ..., xn, yn, zn] where n = (element_node_count-1)
        for (int i = 0; i < coords_per_element; ++i){
            element_node_coords[i] = mesh_node_coords_ptr[element_min_index + i];
        }

        // Find centroid for this element
        compute_element_centroid(&element_node_coords[0], element_centroid, nodes_per_element);
        mesh_element_centroids.push_back(element_centroid);

        // Create bounding volume for this element
        AABB element_aabb;
        element_aabb.build_for_element(&element_node_coords[0], nodes_per_element);
        mesh_element_aabbs.push_back(element_aabb);
        //std::cout << "AABB max " << element_aabb.corner_max[0] << " " << element_aabb.corner_max[1] << " " << element_aabb.corner_max[2] << std::endl;
        //std::cout << "AABB min " << element_aabb.corner_min[0] << " " << element_aabb.corner_min[1] << " " << element_aabb.corner_min[2] << std::endl;

        // Include element AABB in mesh AABB to get the bounding box for the whole mesh
        mesh_aabb.expand_to_include_AABB(element_aabb);
        } // ELEMENTS
    }

AABB create_node_AABB(const std::vector<AABB>& mesh_element_abbs,
    const std::vector<int>& mesh_element_indices,
    const int node_min_element_idx,
    const int node_element_count);

inline double find_SAH_cost_bin(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds);

inline double find_SAH_cost_bin_full(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds,
    const AABB& parent_bounds);

inline void midpoint_split(AABB& node_centroid_bounds,
    double axis_extent,
    unsigned int& out_split_axis,
    double& out_split_position);

bool binned_SAH_split(BuildTask& Node,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    const std::vector<int>& mesh_element_indices,
    unsigned int& out_split_axis,
    double& out_split_position);

bool split_BVH_node(BuildTask &task,
    const std::vector<std::array<double,3>>& element_centroids,
    const std::vector<AABB>& element_aabbs,
    std::vector<int>& element_indices,
    int& out_left_min_element_idx,
    size_t& out_left_count);

void build_BLAS(BLAS &mesh_bvh,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    size_t mesh_element_count);

void build_TLAS(std::vector<TLAS_Node>& TLAS,
    const std::vector<std::array<double,3>>& scene_blas_centroids,
    const std::vector<AABB>& scene_blas_aabbs,
    std::vector<int>& scene_blas_indices,
    size_t scene_mesh_count);

void copy_data_to_BLAS_node_tex(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_uvs_ptr,
    const int mesh_material,
    const int timestep);

void copy_data_to_BLAS_node_color(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_face_color_ptr,
    const int mesh_material,
    const int timestep);

void copy_data_to_TLAS(TLAS &tlas,
    std::vector<BLAS>& scene_BLASes,
    const std::vector<int>& scene_blas_indices);

inline void set_BLAS_material(BLAS &mesh_bvh, const int mesh_material, const double mesh_ri, const double scene_ri, const enum ObjectType mesh_object_type);
inline void set_BLAS_intersection_texture(BLAS &mesh_bvh,  const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type);
inline void set_BLAS_intersection_color(BLAS &mesh_bvh,  const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type);


TLAS build_acceleration_structures(const std::vector <nanobind::ndarray<const double,nanobind::c_contig>>& scene_coords_expanded,
    const std::vector <nanobind::ndarray<const double,nanobind::c_contig>>& scene_normals_expanded,
    const std::vector<nanobind::ndarray<const double,nanobind::c_contig>>& scene_face_colors,
    const std::vector<int>& materials,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_uvs,
    const std::vector<nanobind::ndarray<const double, nanobind::c_contig>>& scene_textures,
    const std::vector<int>& scene_surface_types,
    const std::vector<double>& scene_refractive_indices,
    const std::vector<int>& mesh_priorities,
    const std::vector<int>& mesh_object_types,
    const std::vector<double>& scene_mesh_thickness,
    const int shading_type,
    const int timestep,
    const int timestep_count);


