// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef RTBVH_H
#define RTBVH_H

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

// ================================================================================
// Geometry and BVH data structures
// ================================================================================

/**
 * @brief Axis-aligned bounding box (AABB) for 3D geometry.
 *
 * Stores minimum and maximum corners in Cartesian coordinates and provides
 * helper methods used throughout BVH construction (per-element bounds,
 * centroid bounds, surface area, diagonal length, etc.).
 *
 * Struct size: 2 x 8 x 3 = 48 bytes.
 */
struct AABB {
    double corner_min[3]{};
    double corner_max[3]{};

    /**
     * @brief Constructs an empty AABB initialised to infinite extent.
     *
     * Minimum corner is set to +infinity and maximum corner to -infinity so that subsequent expansion operations initialise it correctly.
     */
    AABB() {
        corner_min[0] = corner_min[1] = corner_min[2] = std::numeric_limits<double>::infinity();
        corner_max[0] = corner_max[1] = corner_max[2] = -std::numeric_limits<double>::infinity();
    }

    // Used for building AABBs for all mesh elements, regardless of the type
    /**
     * @brief Builds an AABB for a single mesh element from its node coordinates.
     *
     * Iterates over all element nodes and updates this bounding box to tightly enclose the element.
     * Works for all supported mesh element types.
     *
     * @param[in] element_node_coords (const double*) Pointer to nodal coordinates laid out as [x0,y0,z0,x1,y1,z1,...]
     * @param[in] element_node_count (int) Number of nodes in the element
     */
    inline void build_for_element(const double* element_node_coords,
        const int element_node_count){
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

     /**
     * @brief Expands this AABB to include a single 3D point.
     *
     * Used when computing centroid bounds for SAH splitting.
     *
     * @param[in] point (const std::array<double,3>&) Point to include
     */
    inline void expand_to_include_point(const std::array<double,3>& point){
        // Iterate over x, y, z coordinates
        for (int i = 0; i < NODE_COORDINATES; ++i){
            double point_coordinate = point[i];
            corner_min[i] = std::min(corner_min[i], point_coordinate);
            corner_max[i] = std::max(corner_max[i], point_coordinate);
        }
    }

    /**
     * @brief Expands this AABB to include another AABB.
     *
     * Used for constructing parent node bounds from child or element AABBs.
     *
     * @param[in] other (const AABB&) Bounding box to merge into this one
     */
    inline void expand_to_include_AABB(const AABB& other) {
        // Iterate over (x, y, z) coordinates
        for (int i = 0; i < NODE_COORDINATES; ++i){
            corner_min[i] = std::min(corner_min[i], other.corner_min[i]);
            corner_max[i] = std::max(corner_max[i], other.corner_max[i]);
        }
    }

    /**
     * @brief Returns the extent (length) of the AABB along a given axis.
     *
     * @param[in] axis (int) Axis index (0 = x, 1 = y, 2 = z)
     *
     * @return (double) Non-negative extent along the specified axis
     */
    inline double find_axis_extent(int axis) const {
        double result = corner_max[axis] - corner_min[axis];
        if (result < 0) return 0.0;
        return result;
    }

    /**
     * @brief Computes the surface area of the AABB.
     *
     * Treats the AABB as a rectangular prism and returns 2*(hw + wd + hd),
     * where h, w and d are the extents in z, y and x directions respectively.
     *
     * @return (double) Surface area of the AABB
     */
    inline double find_surface_area() const {
        double height = find_axis_extent(2);
        double width = find_axis_extent(1);
        double depth = find_axis_extent(0);
        // Surface area of rectangular prism
        return 2 * (height * width + width * depth + height * depth);
    }

    /**
     * @brief Computes the length of the diagonal of the AABB.
     *
     * @return (double) Euclidean distance between minimum and maximum corners.
     */
    inline double find_diagonal() const{
        // Diagonal of rectangular prism
        return std::sqrt(std::pow(corner_max[0] - corner_min[0], 2) + std::pow(corner_max[1] - corner_min[1], 2) + std::pow(corner_max[2] - corner_min[2], 2));
    }
};

/**
 * @brief Bin used for binned SAH splitting in BVH construction.
 *
 * Stores a local bounding box and element count for one SAH bin.
 *
 * Struct size: 48 + 4 = 52 bytes.
 */
struct Bin {
    // Bin for binning SAH
    AABB bounding_box {};
    int element_count {0};
};

/**
 * @brief Temporary data carrier used during BVH construction in build_BLAS and build_TLAS.
 *
 * Encodes which node in the BVH is being processed, how many elements it owns,
 * and the starting index into the global element index array.
 *
 * Struct size: 8 (64-bit) + 4 x 2 = 16 bytes.
 */
struct BuildTask {
    size_t element_count; // Number of elements
    int node_idx;
    int min_element_idx; // First mesh element index in element_indices

    BuildTask() = default;
    /**
     * @brief Constructs a BuildTask with the given parameters.
     *
     * @param[in] element_count (size_t) Number of elements assigned to this node.
     * @param[in] node_idx (int) Index of the BVH node in the node array.
     * @param[in] min_element_idx (int) First element index for this node.
     */
    BuildTask(size_t element_count, int node_idx, int min_element_idx):
        element_count(element_count),
        node_idx(node_idx),
        min_element_idx(min_element_idx)
        {};
};

/**
 * @brief Simple texture descriptor storing a pointer to image data and its width and height.
 *
 * The texture data is kept at the BLAS level and sampled in intersection
 * routines, allowing BLAS nodes to reference it without duplication.
 *
 * Struct size: 8 (64-bit) + 4 x 2 = 16 bytes.
 */
struct Texture {
    const double* data {nullptr}; // Pointer to the texture, so we can just assign it to relevant BVH nodes and sample
    int height {0};
    int width {0};
    
    /// @brief Default constructor initialising an empty texture
   Texture() = default;

   /**
     * @brief Constructs a texture wrapper around existing image data.
     *
     * @param[in] data (const double*) Pointer to the texture buffer.
     * @param[in] height (int) Texture height in pixels.
     * @param[in] width (int) Texture width in pixels.
     */
   Texture(const double* data, int height, int width):
    data(data),
    height(height),
    width(width)
    {};
};

/**
 * @brief BVH node for a Bottom-Level Acceleration Structure (BLAS).
 *
 * 
 * Stores element-local nodal data (coordinates, normals, colors/UVs) together
 * with the node's bounding box and hierarchy information.
 * 
 *
 * In the worst case, each vector contains
 * MAX_ELEMENTS_PER_LEAF * nodes_per_element * 3 doubles.
 *
 * Approximate worst-case size: ~2.65 kB (see comments in source).
 */

 // Struct size in worst case: 
// Each vector: MAX_ELEMENTS_PER_LEAF (currently = 4) x  nodes_per_element (max. 9 for QUAD9) x 3 (x,y,z per node) x 8 (double) = 864 bytes
// Rest: 48 + 8 (64-bit system) + 4 + 1
// Total: 864 x 3 + 48 + 8 + 4 + 1 = 2653 bytes = 2.653 kB
struct BLAS_Node {
    std::vector<double> node_coords; // Coordinates of nodes comprising the mesh elements stored in the node
    std::vector<double> node_normals; // Normals of nodes comprising the mesh elements stored in the node
    std::vector<double> face_color; // Either per-face RGB values ([faces, 3] array) or per-node (u,v) coordinate pairs (textures)
    AABB bounding_box {}; // Bounding box for all elements in this node
    size_t element_count {0}; // Number of elements in this node; if not zero, this is the leaf
    int left_child_idx {-1}; // Index of the left child. Right child is inferred as left_child_idx+1
    enum ElementNodeCount nodes_per_element {ElementNodeCount::TRI3}; // Element type (TRI3, QUAD4, etc.)

    /// @brief Default constructor for use with emplace_back.
   BLAS_Node() = default;
   /**
     * @brief Constructs a BLAS node with given bounding box and element count
     *
     * @param[in] aabb (AABB) Node bounding box.
     * @param[in] element_count (size_t) Number of elements stored in the node
     * @param[in] left_child_idx (int) Index of the left child (-1 for leaf)
     */
   BLAS_Node(AABB aabb, size_t element_count, int left_child_idx):
    bounding_box(aabb),
    element_count(element_count),
    left_child_idx(left_child_idx)
    {}; 
};

// Forward declaration (incomplete types) so we can use them in function pointers in BLAS while avoiding circular dependencies (since they depend on BLAS_Node defined here)
struct IntersectionOutput; 


/**
 * @brief Bottom-level acceleration structure (BLAS) for a single mesh.
 *
 * Holds the BVH tree nodes and all state required for ray–mesh intersection,
 * including per-mesh texture data, material behaviour, and nested volume information.
 *
 */
// Struct size (worst case):
// Vector: number of elements in the mesh * 2653 bytes
// Rest: 48 + 16 + 8 (64-bit system) + 8 x 2 + 8 (64-bit system) x 2 + 4 x 3 = 116 bytes
struct BLAS {
    std::vector<BLAS_Node> tree_nodes; // Flat array of BVH nodes for this mesh
    AABB bounding_box {}; // Bounding box of the entire mesh
    // We cannot just copy relevant pieces of texture into different BLAS nodes, so keep it at the BLAS level. This will also allow us to do fewer if/else checks in intersection for coloring
    Texture texture {}; // If texture.data is not a nullptr, face_color is (u,v). This logic saves us having to store surface type explicitly
    IntersectionOutput (*intersection_function_ptr)(const Ray&, const std::vector<double>& node_coords, const unsigned int bvh_node_element_count) {nullptr}; // Ray-mesh element intersection check (TRI3, QUAD4, etc.) pointer
    double refractive_index {1.0}; // Refractive index of the mesh material; set to 1.0 to avoid bad division in case it somehow gets unitialised
    double thickness {1.0}; // Thickness of a SHELL mesh; unused for solids
    // Void function pointer will be 8 bytes in 64-bit system, 4 in a 32x, so this should be the best positioning of those for memory alignment
    // Function pointer for writing intersection data into a HitRecord. Allows specialising how intersection attributes are computed based on element type and surface representation (solid color, texture, etc.).
    void (*overwrite_intersection_function_ptr)(HitRecord&,const BLAS_Node&, const Texture& texture, Eigen::Index min_row_idx) {nullptr}; 
    // Function pointer for evaluating ray–mesh material interaction
    void (*ray_material_ptr)(const RayState& current_state, HitRecord& intersection_record, const EiVector3d& albedo, std::vector<RayState>& stack, EiVector3d& total_color, const double offset) {nullptr};
    // Uncomment the below 2 lines if deciding to go for switch-based dispatch in return_ray_color
    //ObjectType object_type;
    //int material;
    int priority {0}; // Priority - tells us the ordering of nested volumes in the scene
    int root_idx {-1}; // Index of the root BVH node (usually 0).
    int blas_idx {-1}; // Index in TLAS, used for handling nested refractive volumes without OOP/pointers

    /// @brief Default constructor for use with emplace_back to avoid temporary copies
    BLAS() = default; 
};

/**
 * @brief BVH node in the top-level acceleration structure (TLAS).
 *
 * TLAS nodes group BLAS instances (meshes) into a hierarchy used for coarse-level culling before descending into BLAS trees.
 *
 * Struct size: 48 + 4 x 3 = 60 bytes.
 */
struct TLAS_Node {
    AABB bounding_box {}; // Bounding box for the BLASes in this node
    int blas_count {0}; // Number of BLASes in this node (consecutive in the array)
    int left_child_idx {-1}; // Index of the left child (-1 for leaf)
    int min_blas_idx {-1}; // Index of the first BLAS in this node. Store this instead of data as we expect a few meshes in the scene tops, so indexing into BLAS vector shouldn't be too awful

     // Constructors for emplace_back to avoid temporary copies
    TLAS_Node() = default;
    /**
     * @brief Constructs a TLAS node with given bounding box and BLAS range.
     *
     * @param[in] aabb (AABB) Node bounding box
     * @param[in] count (int) Number of BLASes contained in this node
     * @param[in] left_idx (int) Index of the left child node
     * @param[in] min_blas_idx (int) Index of the first BLAS in this node
     */
    TLAS_Node(AABB aabb, int count, int left_idx, int min_blas_idx):
        bounding_box(aabb),
        blas_count(count),
        left_child_idx(left_idx),
        min_blas_idx(min_blas_idx)
        {};
};

/**
 * @brief Top-level acceleration structure (TLAS) for the entire scene.
 *
 * Owns the TLAS node hierarchy and the BLAS instances ordered according
 * to traversal layout.
 */
struct TLAS {
    std::vector<BLAS> blases; // BLASes stored in TLAS traversal order
    std::vector<TLAS_Node> tlas_nodes; // TLAS BVH nodes
};

// ================================================================================
// Geometric preprocessing helpers
// ================================================================================

/**
 * @brief Computes the centroid of a single mesh element.
 *
 * General helper for any element type, operating on a flat array of nodal coordinates.
 *
 * @param[in] element_node_coords (const double*) Pointer to element nodal coordinates
 *            laid out as [x0,y0,z0,x1,y1,z1,...]
 * @param[out] element_centroid (std::array<double,3>&) Centroid coordinates (x,y,z)
 * @param[in] element_node_count (int) Number of nodes in the element
 */
static inline void compute_element_centroid(const double *element_node_coords, // Pointer to an array, so we can have one centroid function to rule them all without having to specify the array size here
    std::array<double, NODE_COORDINATES> &element_centroid,
    int element_node_count);

/**
 * @brief Computes the centroid of an entire mesh from its AABB.
 *
 * The centroid is taken as the midpoint between minimum and maximum corners
 * of the mesh-level bounding box.
 *
 * @param[in] mesh_aabb (AABB&) Mesh bounding box.
 * @param[out] mesh_centroid (std::array<double,3>&) Mesh centroid (x,y,z).
 */
static inline void compute_mesh_centroid(AABB& mesh_aabb, std::array<double,3>& mesh_centroid);


/**
 * @brief Processes per-element data for a mesh and builds element AABBs.
 *
 * For each element in the mesh, this function:
 *  - gathers nodal coordinates at the specified timestep,
 *  - computes its centroid,
 *  - builds a per-element AABB,
 *  - expands the mesh-level AABB to include it.
 *
 * @tparam element_node_count (ElementNodeCount) Compile-time number of nodes per element
 *
 * @param[in] mesh_number_of_elements (size_t) Number of elements in the mesh
 * @param[in] mesh_node_coords_ptr (const double*) Pointer to mesh nodal coordinates laid out in a time-dependent flattened layout
 * @param[out] mesh_element_centroids (std::vector<std::array<double,3>>&) Per-element centroids
 * @param[out] mesh_element_aabbs (std::vector<AABB>&) Per-element bounding boxes
 * @param[in,out] mesh_aabb (AABB&) Mesh-level bounding box, expanded in-place
 * @param[in] timestep (int) Timestep index for time-dependent meshes at which the data is retrieved
 */
template<ElementNodeCount element_node_count>
static void process_element_data(size_t mesh_number_of_elements,
    const double* mesh_node_coords_ptr,
    std::vector<std::array<double,3>>& mesh_element_centroids,
    std::vector<AABB>& mesh_element_aabbs,
    AABB& mesh_aabb,
    const int timestep){

    // We need these to be compile time constants to create arrays. They are known from switch in build_acceleration_structures
    constexpr int nodes_per_element = static_cast<int>(element_node_count); // Explicitly cast to int
    constexpr int coords_per_element = nodes_per_element * NODE_COORDINATES; // Number of elements times 3 coordinates (x,y,z) each. size_t only to be able to pass this as an argument in array creation
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
        } 
    }

// ================================================================================
// BVH construction and splitting helpers
// ================================================================================

/**
 * @brief Builds an AABB for a BVH node from a subset of element AABBs.
 *
 * Iterates over the elements assigned to the node (via mesh_element_indices) and merges their AABBs into a single node bounding box.
 *
 * @param[in] mesh_element_abbs (const std::vector<AABB>&) Per-element AABBs
 * @param[in] mesh_element_indices (const std::vector<int>&) Permuted element indices
 * @param[in] node_min_element_idx (int) First index in mesh_element_indices for this node
 * @param[in] node_element_count (int) Number of elements assigned to this node
 *
 * @return (AABB) Bounding box enclosing all elements in the node
 */
AABB create_node_AABB(const std::vector<AABB>& mesh_element_abbs,
    const std::vector<int>& mesh_element_indices,
    const int node_min_element_idx,
    const int node_element_count);


/**
 * @brief Computes a simplified Surface Area Heuristic (SAH) cost for a candidate split bin.
 *
 * Uses a basic cost model proportional to element counts and surface areas of the child nodes.
 *
 * @param[in] left_element_count (unsigned int) Number of elements in the left child
 * @param[in] right_element_count (unsigned int) Number of elements in the right child
 * @param[in] left_bounds (const AABB&) Left child bounding box
 * @param[in] right_bounds (const AABB&) Right child bounding box
 *
 * @return (double) Estimated SAH cost for this split.
 */
inline double find_SAH_cost_bin(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds);

/**
 * @brief Computes a full SAH cost for a candidate split bin.
 *
 * Implements a more complete SAH model using traversal cost, intersection cost, and normalised child surface areas relative to the parent node.
 *
 * @param[in] left_element_count (unsigned int) Number of elements in the left child
 * @param[in] right_element_count (unsigned int) Number of elements in the right child
 * @param[in] left_bounds (const AABB&) Left child bounding box
 * @param[in] right_bounds (const AABB&) Right child bounding box
 * @param[in] parent_bounds (const AABB&) Parent node bounding box
 *
 * @return (double) Estimated SAH cost for this split.
 */
inline double find_SAH_cost_bin_full(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds,
    const AABB& parent_bounds);

/**
 * @brief Computes a midpoint split as a fallback when SAH fails.
 *
 * Splits the node at the midpoint of its centroid bounds along the chosen axis.
 *
 * @param[in] node_centroid_bounds (AABB&) Bounding box of element centroids for this node
 * @param[in] axis_extent (double) Extent along the chosen split axis
 * @param[out] out_split_axis (unsigned int&) Index of the split axis
 * @param[out] out_split_position (double&) World-space split position
 */
inline void midpoint_split(AABB& node_centroid_bounds,
    double axis_extent,
    unsigned int& out_split_axis,
    double& out_split_position);

/**
 * @brief Performs binned SAH splitting for a BVH node.
 *
 * Computes centroid bounds, bins elements along the best axis, and evaluates all possible split positions to find the one that minimises SAH cost.
 * Falls back to midpoint splitting if necessary.
 *
 * @param[in] Node (BuildTask&) Build task describing the current node
 * @param[in] mesh_element_centroids (const std::vector<std::array<double,3>>&) Per-element centroids
 * @param[in] mesh_element_aabbs (const std::vector<AABB>&) Per-element AABBs
 * @param[in] mesh_element_indices (const std::vector<int>&) Permutation of element indices for this BLAS
 * @param[out] out_split_axis (unsigned int&) Chosen split axis
 * @param[out] out_split_position (double&) Chosen split position along that axis
 *
 * @return (bool) True if a valid split was found (or midpoint fallback applied), otherwise false.
 */
bool binned_SAH_split(BuildTask& Node,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    const std::vector<int>& mesh_element_indices,
    unsigned int& out_split_axis,
    double& out_split_position);

/**
 * @brief Splits a BVH node into left and right child ranges.
 *
 * Partitions the element index array in-place according to the chosen split position and axis,
 * returning the element range for the left child.
 *
 * @param[in] task (BuildTask&) Build task describing the node to split
 * @param[in] element_centroids (const std::vector<std::array<double,3>>&) Per-element centroids
 * @param[in] element_aabbs (const std::vector<AABB>&) Per-element AABBs
 * @param[in,out] element_indices (std::vector<int>&) Permuted element indices to partition
 * @param[out] out_left_min_element_idx (int&) First index of the left child range
 * @param[out] out_left_count (size_t&) Number of elements in the left child
 *
 * @return (bool) True if a valid split was performed, otherwise false.
 */
bool split_BVH_node(BuildTask &task,
    const std::vector<std::array<double,3>>& element_centroids,
    const std::vector<AABB>& element_aabbs,
    std::vector<int>& element_indices,
    int& out_left_min_element_idx,
    size_t& out_left_count);

// ================================================================================
// BLAS and TLAS builders
// ================================================================================

/**
 * @brief Builds a BLAS (BVH) for a single mesh.
 *
 * Constructs a BVH over mesh elements using SAH-based splitting, and records
 * the minimum element index for each node so per-node data can be copied later.
 *
 * @param[out] mesh_bvh (BLAS&) BLAS to fill for this mesh
 * @param[in] mesh_element_centroids (const std::vector<std::array<double,3>>&) Per-element centroids
 * @param[in] mesh_element_aabbs (const std::vector<AABB>&) Per-element AABBs
 * @param[in,out] mesh_element_indices (std::vector<int>&) Permuted element indices (updated by the builder)
 * @param[out] node_minimum_element_index (std::vector<int>&) Minimum element index for each BVH node
 * @param[in] mesh_element_count (size_t) Number of elements in the mesh
 * @param[in] nodes_per_element (ElementNodeCount) Element type for this mesh
 */
void build_BLAS(BLAS &mesh_bvh,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    size_t mesh_element_count);

/**
 * @brief Builds a TLAS (BVH) over the BLAS instances in the scene.
 *
 * Treats each BLAS as a single element with its own AABB and centroid, and
 * builds a BVH used for coarse pre-liminary intersections
 *
 * @param[out] TLAS_nodes (std::vector<TLAS_Node>&) TLAS node array to fill
 * @param[in] scene_blas_centroids (const std::vector<std::array<double,3>>&) Per-BLAS centroids
 * @param[in] scene_blas_aabbs (const std::vector<AABB>&) Per-BLAS AABBs
 * @param[in,out] scene_blas_indices (std::vector<int>&) Permuted BLAS indices (updated by the builder)
 * @param[in] scene_mesh_count (size_t) Number of meshes (BLASes) in the scene
 */
void build_TLAS(std::vector<TLAS_Node>& TLAS,
    const std::vector<std::array<double,3>>& scene_blas_centroids,
    const std::vector<AABB>& scene_blas_aabbs,
    std::vector<int>& scene_blas_indices,
    size_t scene_mesh_count);

// ================================================================================
// Data transfer into BVH nodes
// ================================================================================

/**
 * @brief Copies mesh geometry and texture data into BLAS leaf nodes (textured surfaces).
 *
 * For each leaf node, this function:
 *  - copies nodal coordinates and normals for the assigned elements,
 *  - copies per-node UV coordinates for texturing.
 *
 * @param[in,out] mesh_bvh (BLAS&) BLAS whose nodes will receive the data
 * @param[in] mesh_element_indices (std::vector<int>&) Permuted element indices
 * @param[in] node_minimum_element_index (std::vector<int>&) Minimum element index per BVH node
 * @param[in] mesh_node_coords_expanded_ptr (const double*) Pointer to nodal coordinates array
 * @param[in] mesh_node_normals_expanded_ptr (const double*) Pointer to nodal normals array
 * @param[in] mesh_uvs_ptr (const double*) Pointer to per-node UV coordinates
 * @param[in] mesh_material (int) Material identifier for the mesh
 * @param[in] timestep (int) Timestep index for time-dependent data
 */
void copy_data_to_BLAS_node_tex(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_uvs_ptr,
    const int mesh_material,
    const int timestep);

/**
 * @brief Copies mesh geometry and per-face color data into BLAS leaf nodes (solid surfaces).
 *
 * For each leaf node, this function:
 *  - copies nodal coordinates and normals for the assigned elements,
 *  - copies per-element RGB values representing face colors or scalar fields.
 *
 * @param[in,out] mesh_bvh (BLAS&) BLAS whose nodes will receive the data
 * @param[in] mesh_element_indices (std::vector<int>&) Permuted element indices
 * @param[in] node_minimum_element_index (std::vector<int>&) Minimum element index per BVH node
 * @param[in] mesh_node_coords_expanded_ptr (const double*) Pointer to nodal coordinates array
 * @param[in] mesh_node_normals_expanded_ptr (const double*) Pointer to nodal normals array
 * @param[in] mesh_face_color_ptr (const double*) Pointer to per-face color values
 * @param[in] mesh_material (int) Material identifier for the mesh
 * @param[in] timestep (int) Timestep index for time-dependent data
 */
void copy_data_to_BLAS_node_color(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_face_color_ptr,
    const int mesh_material,
    const int timestep);

/**
 * @brief Copies BLAS objects into the TLAS in traversal order.
 *
 * Reorders BLAS instances into the TLAS according to the BLAS index permutation determined by the TLAS builder.
 *
 * @param[in,out] tlas (TLAS&) TLAS to populate with BLAS instances
 * @param[in] scene_BLASes (std::vector<BLAS>&) Original BLAS list
 * @param[in] scene_blas_indices (const std::vector<int>&) Permutation mapping into scene_BLASes
 */
void copy_data_to_TLAS(TLAS &tlas,
    std::vector<BLAS>& scene_BLASes,
    const std::vector<int>& scene_blas_indices);

// ================================================================================
// BLAS material and shading configuration
// ================================================================================

/**
 * @brief Configures material behaviour for a BLAS.
 *
 * Sets the material-specific ray interaction function and assigns the
 * appropriate refractive index based on mesh and scene media.
 *
 * @param[in,out] mesh_bvh (BLAS&) BLAS corresponding to the mesh
 * @param[in] mesh_material (int) Material identifier (e.g. DIFFUSE, SPECULAR)
 * @param[in] mesh_ri (double) Refractive index of the mesh material
 * @param[in] scene_ri (double) Refractive index of the surrounding medium
 * @param[in] mesh_object_type (ObjectType) Object type (e.g. SOLID or SHELL)
 */
inline void set_BLAS_material(BLAS &mesh_bvh, const int mesh_material, const double mesh_ri, const double scene_ri, const enum ObjectType mesh_object_type);

/**
 * @brief Selects the intersection overwrite function for textured meshes.
 *
 * Assigns a function pointer for writing intersection data (UVs, normals, etc.)
 * into the HitRecord based on element type and shading mode.
 *
 * @param[in,out] mesh_bvh (BLAS&) BLAS to configure
 * @param[in] nodes_per_element (ElementNodeCount) Element type for this mesh
 * @param[in] shading_type (ShadingType) Shading mode (flat, blended, etc.)
 */
inline void set_BLAS_intersection_texture(BLAS &mesh_bvh,  const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type);

/**
 * @brief Selects the intersection overwrite function for solid-colored meshes.
 *
 * Assigns a function pointer for writing intersection data (colors, normals, etc.)
 * into the HitRecord based on element type and shading mode.
 *
 * @param[in,out] mesh_bvh (BLAS&) BLAS to configure
 * @param[in] nodes_per_element (ElementNodeCount) Element type for this mesh
 * @param[in] shading_type (ShadingType) Shading mode (flat, blended, etc.)
 */
inline void set_BLAS_intersection_color(BLAS &mesh_bvh,  const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type);

// ================================================================================
// High-level builder
// ================================================================================

/**
 * @brief Builds BLAS and TLAS acceleration structures for an entire scene.
 *
 * For each mesh in the scene this function:
 *  - extracts nodal coordinates and normals from Python buffers,
 *  - computes per-element centroids and AABBs,
 *  - builds a BLAS using SAH-based BVH construction,
 *  - assigns material and shading behaviour and copies per-node data
 *    into BLAS leaf nodes (color or texture),
 *  - accumulates per-mesh centroids and AABBs.
 *
 * After processing all meshes, it:
 *  - builds a TLAS over the BLAS instances,
 *  - copies BLASes into TLAS storage in traversal order,
 *  - returns the complete TLAS for use in ray tracing.
 *
 * @param[in] scene_coords_expanded (const std::vector<nanobind::ndarray>&)
 *            Time-dependent nodal coordinates for each mesh
 * @param[in] scene_normals_expanded (const std::vector<nanobind::ndarray>&)
 *            Time-dependent nodal normals for each mesh
 * @param[in] scene_face_colors (const std::vector<nanobind::ndarray>&)
 *            Per-face color data for solid meshes
 * @param[in] materials (const std::vector<int>&)
 *            Material identifiers per mesh
 * @param[in] scene_uvs (const std::vector<nanobind::ndarray>&)
 *            Per-node UV coordinates for textured meshes
 * @param[in] scene_textures (const std::vector<nanobind::ndarray>&)
 *            Texture images for textured meshes
 * @param[in] scene_surface_types (const std::vector<int>&)
 *            Surface type flags (e.g. solid vs textured) per mesh
 * @param[in] scene_refractive_indices (const std::vector<double>&)
 *            Refractive indices for each mesh and the scene medium
 * @param[in] mesh_priorities (const std::vector<int>&)
 *            Nested-volume priorities per mesh
 * @param[in] mesh_object_types (const std::vector<int>&)
 *            Object type identifiers per mesh (cast to ObjectType)
 * @param[in] scene_mesh_thickness (const std::vector<double>&)
 *            Shell thickness per mesh
 * @param[in] shading_type (int)
 *            Shading mode identifier (cast to ShadingType)
 * @param[in] timestep (int)
 *            Timestep index to build structures for
 * @param[in] timestep_count (int)
 *            Total number of timesteps in the scene
 *
 * @return (TLAS) Fully built top-level acceleration structure for the scene.
 */
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


#endif // RTBVH_H