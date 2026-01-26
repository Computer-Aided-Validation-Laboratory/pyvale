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

// Enum storing the number of nodes per element, so we can update it later nicely when we add different types? At least that's the idea.
enum ElementNodeCount {
    TRI3 = 3,
    QUAD4 = 4,
    TET4 = 4,
    TET10 = 10,
    TET14 = 14,
    HEX8 = 8,
    HEX20 = 20,
    HEX27 = 27
};

// Bounding volume structure - axis-aligned bounding boxes (AABB)
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }

    // Used for building AABBs for all mesh triangles
    inline void build_for_tri3(const std::array<double,9> &triangle_node_coords){
        for (int node = 0; node < 3; ++node) {
        const int offset = node * 3;
            for (int i = 0; i < 3; ++i) {
                const double nodal_coordinate = triangle_node_coords[offset + i];
                corner_min[i] = std::min(corner_min[i], nodal_coordinate);
                corner_max[i] = std::max(corner_max[i], nodal_coordinate);
            }
        }
    }   

     // Used for SAH splitting
    inline void expand_to_include_point(const std::array<double,3>& point){
        for (int i = 0; i < 3; ++i){
            double point_coordinate = point[i];
            corner_min[i] = std::min(corner_min[i], point_coordinate);
            corner_max[i] = std::max(corner_max[i], point_coordinate);
        }
    }
     // Used for creating child node AABBs
    inline void expand_to_include_AABB(const AABB& other) {
        for (int i = 0; i < 3; ++i){
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
        return 2 * (height * width + width * depth + height * depth);
    }
};

struct Bin {
    // Bin for binning SAH
    AABB bounding_box {};
    int element_count {0};
};

// Struct used as a temporary data carrier in build_BLAS and build_TLAS
struct BuildTask {
    size_t element_count;      // number of elements
    int node_idx;
    int min_element_idx;      // first triangle index in tri_indices
};

// BLAS - Bottom Level Acceleration Structure. Each BLAS stores a BVH for one mesh in the scene
struct BLAS_Node {
    std::vector<double> node_coords; // Coordinates of nodes comprising the mesh elements stored in the node, if applicable
    std::vector<double> face_color; // Element (face) colors based on the field values for the mesh
    AABB bounding_box {};
    size_t element_count {0}; // If not zero, this is the leaf
    enum ElementNodeCount nodes_per_element {TRI3}; // Assign 3 by default for now since we only do triangles
    int left_child_idx {-1};
   // int right_child_idx {-1}; // Removed as it's left + 1, but helpful to keep it here for debugging
    //int min_elem_idx {-1};

    // Constructors for emplace_back to avoid temporary copies
   BLAS_Node() = default;
   BLAS_Node(AABB aabb, size_t element_count, int left_child_idx):
    bounding_box(aabb),
    element_count(element_count),
    left_child_idx(left_child_idx)
    {}; 
};

struct BLAS {
    std::vector<BLAS_Node> tree_nodes;
    AABB bounding_box;
    int root_idx {-1};

    BLAS() = default; // Constructor for emplace_back to avoid temporary copies
};

// TLAS - Top Level Acceleration Structure. Stores all BLASes for the scene, used for preliminary intersection
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

inline void compute_element_centroid_tri3(const std::array<double,9> &triangle_node_coords,
    std::array<double,3> &triangle_centroid);

inline void compute_mesh_centroid(AABB& mesh_aabb, std::array<double,3>& mesh_centroid);

void process_element_data_tri3(int mesh_number_of_triangles,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_triangle_aabbs,
    AABB& mesh_aabb,
    const int timestep);

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

void copy_data_to_BLAS_node(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_face_color_ptr,
    const int timestep);
    
void copy_data_to_TLAS(TLAS &tlas,
    std::vector<BLAS>& scene_BLASes,
    const std::vector<int>& scene_blas_indices);

TLAS build_acceleration_structures(const std::vector <nanobind::ndarray<const double,nanobind::c_contig>>& scene_coords_expanded,
    const std::vector<nanobind::ndarray<const double,nanobind::c_contig>>& scene_face_colors,
    const int timestep,
    const int timestep_count);
