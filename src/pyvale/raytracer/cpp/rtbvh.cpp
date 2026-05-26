// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>

// common_cpp header files
#include "../../common_cpp/Eigen/Dense"

// raytracer header files
#include "rtbvh.h"
#include "rtrayintersection.h"
#include "rthitrecord.h"

inline void compute_element_centroid(const double *element_node_coords,
    std::array<double, NODE_COORDINATES> &element_centroid,
    int element_node_count){
    // General function finding the centroid for any element type

    element_centroid.fill(0.0);
    // Iterate over all nodes and sum up their respective x,y,z values
    // node_coords is structured as [x0, y0, z0, x1, y1, z1, ..., xn, yn, zn] where n = (element_node_count-1)
    for (int i = 0; i < element_node_count; ++i){
        element_centroid[0] += element_node_coords[i * NODE_COORDINATES]; // x-component
        element_centroid[1] += element_node_coords[i * NODE_COORDINATES + 1]; // y-component
        element_centroid[2] += element_node_coords[i * NODE_COORDINATES + 2]; // z-component
    }
    // Divide by the number of nodes to get the average
    element_centroid[0] /= element_node_count;
    element_centroid[1] /= element_node_count;
    element_centroid[2] /= element_node_count;
}

inline void compute_mesh_centroid(AABB& mesh_aabb, std::array<double,3>& mesh_centroid) {
    // Compute centroid of the mesh AABB
    for (int i = 0; i < NODE_COORDINATES; ++i){
        mesh_centroid[i] = (mesh_aabb.corner_min[i] + mesh_aabb.corner_max[i]) / 2.0;
    }
}



AABB create_node_AABB(const std::vector<AABB>& mesh_element_abbs,
    const std::vector<int>& mesh_element_indices,
    const int node_min_element_idx,
    const int node_element_count) {
    // Iterates over all elements assigned to the BVH node to find its bounding box
    int node_max_element_idx = node_min_element_idx + node_element_count;
    AABB node_AABB;

    for (int i = node_min_element_idx; i < node_max_element_idx; ++i) {
        int element_idx = mesh_element_indices[i];
        node_AABB.expand_to_include_AABB(mesh_element_abbs[element_idx]);
    }
    return node_AABB;
}

// Auxiliary functions for splitting and binning
inline double find_SAH_cost_bin(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds) {
    // Calculate the Surface Area Heuristic (SAH) cost of a bin (portion of a BVH node). Simplified equation for initial implementation.
    double left_count = static_cast<double>(left_element_count);
    double right_count = static_cast<double>(right_element_count);
    return left_count * left_bounds.find_surface_area() + right_count * right_bounds.find_surface_area(); // Static casts complained so leave C-style casts for now
}

// Full SAH implementation outline, but need to come up with some sensible values for parameters
inline double find_SAH_cost_bin_full(unsigned int left_element_count,
    unsigned int right_element_count,
    const AABB& left_bounds,
    const AABB& right_bounds,
    const AABB& parent_bounds){
    
    double cost_traversal = 1.0;
    double cost_intersection = 2.0; // Might have to vary this with the element type when we start using more than just triangles
    double area_parent = parent_bounds.find_surface_area();
    double area_left_child = left_bounds.find_surface_area();
    double area_right_child = right_bounds.find_surface_area();
    return cost_traversal + cost_intersection * (area_left_child/area_parent * left_element_count + area_right_child/area_parent * right_element_count);
}

inline void midpoint_split(AABB& node_centroid_bounds,
    double axis_extent,
    unsigned int& out_split_axis,
    double& out_split_position){
    // Fallback splitting if SAH fails: midpoint
    
    std::cout << "SAH splitting failed. Trying midpoint instead." << std::endl;
    out_split_position = node_centroid_bounds.corner_min[out_split_axis] + axis_extent * 0.5;
}

bool binned_SAH_split(BuildTask& Node,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    const std::vector<int>& mesh_element_indices,
    unsigned int& out_split_axis,
    double& out_split_position) {
    // Binned Surface Area Heuristic (SAH) split

    if (Node.element_count <= 2) return false; // Too small to split
    unsigned int node_max_element_idx = Node.min_element_idx + Node.element_count;

    // Compute centroid bounds for the node
    // We use existing AABB since it nicely implements everything we need, BUT it is not to be confused with the actual bounding box of the node
    // node_centroid_bounds - Only used to determine splitting
    // bounding_box - Actual bounding box of the node used for ray intersections
    AABB node_centroid_bounds{};
    for (int i = Node.min_element_idx; i < node_max_element_idx; ++i) {
        // Retrieve triangle and its centroid on the split axis
        unsigned int element_idx = mesh_element_indices[i];
        std::array<double,3> element_centroid = mesh_element_centroids[element_idx];
        node_centroid_bounds.expand_to_include_point(element_centroid);
    }

    // Pick the longest axis for splitting
    int best_axis = 0;
    double axis_extent = node_centroid_bounds.find_axis_extent(best_axis);
    for (int i = 1; i < 3; ++i) {
        double temp_extent = node_centroid_bounds.find_axis_extent(i);
        if (temp_extent > axis_extent){
            best_axis = i;
            axis_extent = temp_extent;
        }
    }
    out_split_axis = best_axis;
    //midpoint_split(node_centroid_bounds, axis_extent, out_split_axis, out_split_position); // Uncomment to test midpoint directly

    // All centroids coincident along the chosen axis => No useful split
    if (axis_extent == 0){
        //return false;
        midpoint_split(node_centroid_bounds, axis_extent, out_split_axis, out_split_position);
        return true;
    }

    // Create bins
    constexpr int NUM_BINS = 8;
    Bin bins[NUM_BINS];

    const double inverse_extent = 1.0/axis_extent;
    for (unsigned int i = Node.min_element_idx; i < node_max_element_idx; ++i){
        unsigned int element_idx = mesh_element_indices[i];
        // Find the Bin containing the triangle centroid
        double t = (mesh_element_centroids[element_idx][best_axis] - node_centroid_bounds.corner_min[best_axis]) * inverse_extent;
        int bin_id = static_cast<int>(t * NUM_BINS);
        if (bin_id == NUM_BINS) bin_id = NUM_BINS - 1; // Round up to the last Bin
        bins[bin_id].element_count++;
        bins[bin_id].bounding_box.expand_to_include_AABB(mesh_element_aabbs[element_idx]);
    }

    // Pre-compute left/right bounds for all possible splits (so we don't have to recompute them from scratch to analyse every possible split)
    unsigned int left_count[NUM_BINS], right_count[NUM_BINS];
    AABB left_bounds[NUM_BINS], right_bounds[NUM_BINS];

    // Left-to-right
    AABB possible_left_box;
    unsigned int possible_left_count = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (bins[i].element_count > 0) {
            possible_left_box.expand_to_include_AABB(bins[i].bounding_box);
        }
        possible_left_count += bins[i].element_count;
        left_bounds[i] = possible_left_box;
        left_count[i] = possible_left_count;
    }
    // Right-to-left
    AABB possible_right_box;
    unsigned int possible_right_count = 0;
    for (int i = NUM_BINS - 1; i >= 0; --i) {
        if (bins[i].element_count > 0) {
            possible_right_box.expand_to_include_AABB(bins[i].bounding_box);
        }
        possible_right_count += bins[i].element_count;
        right_bounds[i] = possible_right_box;
        right_count[i] = possible_right_count;
        if (i == 0) break; // Safety
    }

    // Evaluate SAH at each Bin boundary and pick the best one (i.e., the one which minimizes the cost function)
    double best_cost = std::numeric_limits<double>::infinity();
    int best_split_bin = -1;
    AABB parent_node_aabb = create_node_AABB(mesh_element_aabbs, mesh_element_indices, Node.min_element_idx, Node.element_count);

    for (int i = 0; i < NUM_BINS - 1; ++i) {
        unsigned int left_size = left_count[i];
        unsigned int right_size = right_count[i+1];
        if (left_size == 0 || right_size == 0) continue; // invalid split

        // Simplified or full SAH cost calculation
        double cost = find_SAH_cost_bin_full(left_size, right_size, left_bounds[i], right_bounds[i+1], parent_node_aabb);
        //double cost = find_SAH_cost_bin(left_size, right_size, left_bounds[i], right_bounds[i+1]);
        if (cost < best_cost) {
            best_cost = cost;
            best_split_bin = i;
        }
    }
    if (best_split_bin == -1){ // No useful split found
        //return false; 
        midpoint_split(node_centroid_bounds, axis_extent, out_split_axis, out_split_position);
        return true;
    } 

    // Convert Bin index to world-space split position
    double bin_width = axis_extent / NUM_BINS;
    out_split_position = node_centroid_bounds.corner_min[best_axis] + bin_width * (best_split_bin + 1); // Boundary between best_split_bin and best_split_bin + 1
    
    return true;
}

bool split_BVH_node(BuildTask &task,
    const std::vector<std::array<double,3>>& element_centroids,
    const std::vector<AABB>& element_aabbs,
    std::vector<int>& element_indices,
    int& out_left_min_element_idx,
    size_t& out_left_count){

    // Splitting BVH node into child nodes
    // Here element = face in case of a BLAS, or element = BLAS in case of a TLAS
    int element_count = task.element_count;
    unsigned int min_element_idx = out_left_min_element_idx;

    // Run binned SAH
    unsigned int split_axis = 0;
    double split_position = 0.0;
    bool found_split = binned_SAH_split(task, element_centroids, element_aabbs, element_indices, split_axis, split_position);
    if (!found_split) {
        // Fallback splitting implemented, so if SAH returns false, it ought to be too small to split => Mark as leaf node.
        return false;
    }
            
    // Partition of indices by centroid[axis] < split_pos. A bit like QuickSort partitioning, where we get the pivot from our splitting function
    unsigned int begin = min_element_idx;
    unsigned int end = begin + element_count;
    unsigned int mid = begin;

    while (mid < end) {
        unsigned int element_idx = element_indices[mid];
        double element_centroid_split = element_centroids[element_idx][split_axis];
        // Compare triangle centroid position on the axis versus the splitting point
        if (element_centroid_split < split_position) { // Triangle on the left
            ++mid; // move mid to the right
        } else {
            --end; // Move end to left
            std::swap(element_indices[mid], element_indices[end]);
        }
    }
    // How many elements are on the left and on the right
    size_t left_count = mid - begin;
    size_t right_count = element_count - left_count;
    out_left_count = left_count;
    out_left_min_element_idx = begin;
        
    // Abort split if one side is empty
    // NOTE: this could be improved by going through the midpoint split again, but choosing a different axis.
    // That being said, with both SAH and midpoint splitting, this is very unlikely
    if (left_count == 0 || right_count == 0) {
        return false;
    }
    return true;
}

void build_BLAS(BLAS &mesh_bvh,
    const std::vector<std::array<double,3>>& mesh_element_centroids,
    const std::vector<AABB>& mesh_element_aabbs,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    size_t mesh_element_count,
    enum ElementNodeCount nodes_per_element){

    // std::cout << nodes_per_element << '\n';

    // DEBUG HINT: If your render isn't correct and you want to test the intersection without potential influences from the BVH, set MAX_ELEMENT_PER_LEAF
    // to mesh_element_count (just noting that you either have to read and hardcode the value or change type from constexpr)
    static constexpr int MAX_ELEMENTS_PER_LEAF = 4; // Max number of mesh faces per leaf node. According to research 4-16 range works best
    //int MAX_ELEMENTS_PER_LEAF = mesh_element_count; // Max number of mesh faces per leaf node. According to research 4-16 range works best

    // DFS implementation so LIFO
    mesh_bvh.tree_nodes.clear();
    mesh_bvh.tree_nodes.reserve(mesh_element_indices.size() * 2); // crude upper bound

    //std::cout << "BLAS builder: Splitting into nodes..." << std::endl;
  
    // Create root
    BLAS_Node root;
    root.nodes_per_element = nodes_per_element;
    root.element_count = mesh_element_count;
    root.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, 0, mesh_element_count);
    //root.min_elem_idx = 0;
    mesh_bvh.tree_nodes.push_back(root);
    node_minimum_element_index.push_back(0);
    mesh_bvh.root_idx = 0;

    //std::cout << "Initializing building BVH" << std::endl;
    // Stack-based builder
    std::vector<BuildTask> stack;
    stack.push_back({root.element_count, mesh_bvh.root_idx, 0}); // push root onto the stack
   
    while(!stack.empty()){
        //std::cout << "Inside loop for building BVH" << std::endl;
        BuildTask task = stack.back(); // Get address to the last element on the stack
        stack.pop_back(); // Remove the last element from the stack
        int node_idx = task.node_idx;
        int min_element_idx = task.min_element_idx;
        int element_count = task.element_count;
        BLAS_Node& Node = mesh_bvh.tree_nodes[node_idx];
        Node.nodes_per_element = nodes_per_element;

         // Check if we should terminate and make a leaf node
        if (element_count <= MAX_ELEMENTS_PER_LEAF) {
            // Leaf node means that both children indices are -1, so while these should be default values, set them again just to be sure
            Node.element_count = element_count;
            Node.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, min_element_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Split into child nodes
        int left_min_element_idx = min_element_idx;
        size_t left_count = 0;

        if (!split_BVH_node(task, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, left_min_element_idx, left_count)) {
            // Splitting node failed -> Either no split found or we have 0 elements on either right or left side.
            // Fallbacks implemented in the above function, so if it fails, just create leaf node
            Node.element_count = element_count;
            Node.bounding_box = create_node_AABB(mesh_element_aabbs, mesh_element_indices, min_element_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Create children
        int left_child_idx = mesh_bvh.tree_nodes.size();
        int right_child_idx = left_child_idx + 1;
        // Assign element ranges
        // Left child indices: [begin, begin+left_count) <- this is done in split_bvh_node
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        size_t right_count = element_count - left_count;
        int right_min_element_idx = left_min_element_idx + left_count;
        node_minimum_element_index.push_back(left_min_element_idx);
        node_minimum_element_index.push_back(right_min_element_idx);
         
        // Create left child directly in BVH
        mesh_bvh.tree_nodes.emplace_back(create_node_AABB(mesh_element_aabbs, mesh_element_indices, left_min_element_idx, left_count),
            left_count,
            -1);
         // Create right child directly in BVH
        mesh_bvh.tree_nodes.emplace_back(create_node_AABB(mesh_element_aabbs, mesh_element_indices, right_min_element_idx, right_count),
            right_count,
            -1);

         // Set parent data
        // This way instead of using references, as if the vector resizes when we add children, the references might become invalid and produce nonsensical results
        mesh_bvh.tree_nodes[node_idx].left_child_idx = left_child_idx;
        mesh_bvh.tree_nodes[node_idx].element_count = 0; // It is now an internal node
        
        // Push children to stack. LIFO -> Left child gets processed first
        stack.push_back({right_count, right_child_idx, right_min_element_idx});
        stack.push_back({left_count, left_child_idx, left_min_element_idx});
    }
}

void build_TLAS(std::vector<TLAS_Node>& TLAS,
    const std::vector<std::array<double,3>>& scene_blas_centroids,
    const std::vector<AABB>& scene_blas_aabbs,
    std::vector<int>& scene_blas_indices,
    size_t scene_mesh_count){

    static constexpr int MAX_ELEMENTS_PER_LEAF = 2; // Max number of BLASes per leaf node. Meshes are big, so more than 2 doesn't seem to make sense

    // DFS implementation so LIFO
    //std::cout << "TLAS builder: Splitting into nodes..." << std::endl;

    // Create root
    TLAS_Node root;
    root.blas_count = scene_mesh_count;
    root.min_blas_idx = 0;
    root.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, 0, root.blas_count);
    TLAS.push_back(root);

    // Stack-based builder
    std::vector<BuildTask> stack;
    stack.push_back({scene_mesh_count, 0, 0});
   
    while(!stack.empty()){
        //std::cout << "Inside loop for building BVH" << std::endl;
        BuildTask task = stack.back(); // Get address to the last element on the stack
        stack.pop_back(); // Remove the last element from the stack
        int node_idx = task.node_idx;
        int min_blas_idx = task.min_element_idx;
        int element_count = task.element_count;
        TLAS_Node& Node = TLAS[node_idx];

         // Check if we should terminate and make a leaf node
        if (element_count <= MAX_ELEMENTS_PER_LEAF) {
            // Leaf node means that both children indices are -1, so while these should be default values, set them again just to be sure
            Node.min_blas_idx = min_blas_idx;
            Node.blas_count = element_count;
            Node.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, min_blas_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Split into child nodes
        int left_min_element_idx = min_blas_idx;
        size_t left_count = 0;

        if (!split_BVH_node(task, scene_blas_centroids, scene_blas_aabbs, scene_blas_indices, left_min_element_idx, left_count)) {
            // Splitting node failed -> Either no split found or we have 0 elements on either right or left side.
            // Fallbacks implemented in the above function, so if it fails, just create leaf node
            Node.min_blas_idx = min_blas_idx;
            Node.blas_count = element_count;
            Node.bounding_box = create_node_AABB(scene_blas_aabbs, scene_blas_indices, min_blas_idx, element_count);
            Node.left_child_idx = -1;
            continue;
        }

        // Create children
        int left_child_idx = TLAS.size();
        int right_child_idx = left_child_idx + 1;
         // Assign element ranges
        // Left child indices: [begin, begin+left_count)  <- this is done in split_bvh_node
        // Right child indices: [begin+left_count, begin+left_count+right_count)
        size_t right_count = element_count - left_count;
        int right_min_element_idx = left_min_element_idx + left_count;

        // Create left child directly in TLAS
        TLAS.emplace_back(create_node_AABB(scene_blas_aabbs, scene_blas_indices, left_min_element_idx, left_count),
            left_count,
            -1,
            left_min_element_idx);
        // Create right child directly in TLAS
            TLAS.emplace_back(create_node_AABB(scene_blas_aabbs, scene_blas_indices, right_min_element_idx, right_count),
            right_count,
            -1,
            right_min_element_idx);

         // Set parent data
        // This way instead of using references, as if the vector resizes when we add children, the references might become invalid and produce nonsensical results
        TLAS[node_idx].left_child_idx = left_child_idx;
        TLAS[node_idx].blas_count = 0; // It is now an internal node
        
        // Push children to stack. LIFO -> Left child gets processed first
        stack.push_back({right_count, right_child_idx, right_min_element_idx});
        stack.push_back({left_count, left_child_idx, left_min_element_idx});
    }
}

void copy_data_to_BLAS_node_tex(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_uvs_ptr,
    const int mesh_material,
    const int timestep){
    // Texture version - UVs are accessed slightly differently than solid color values
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection and be cache-friendly
    // This way we also avoid copying the mesh data when we move the node to the BVH tree vector as they're already there when we get to this part here.

    //std::cout << "BLAS builder: Copying mesh data into leaf nodes..." << std::endl;
    size_t bvh_node_count = mesh_bvh.tree_nodes.size();
    int mesh_element_count = mesh_element_indices.size();
   
    // Iterate over all BVH nodes
    for (int i = 0; i < bvh_node_count; ++i){
        BLAS_Node& Node = mesh_bvh.tree_nodes[i];

        // std::cout << Node.nodes_per_element << '\n';
        
        // Get indices of the mesh elements assigned to the node for the for loop 
        const int node_min_element_idx = node_minimum_element_index[i];
        const int node_element_count = Node.element_count;
        const int node_max_element_idx = node_min_element_idx + Node.element_count;
        const int coords_per_element = Node.nodes_per_element * NODE_COORDINATES; // number of nodes per element times 3 coordinates each
        const int uvs_per_element = Node.nodes_per_element * UV_COORDINATES; // number of nodes per element times 2 coordinates each
        Node.node_coords.reserve(node_element_count * coords_per_element);
        Node.node_normals.reserve(node_element_count * coords_per_element);
        Node.face_color.reserve(node_element_count * Node.nodes_per_element * UV_COORDINATES); // face_color will store uvs; each comprising vertex/node will have its own uvs
        //Node.material.reserve(node_element_count * NODE_COORDINATES);

        //std::cout << "BVH node id: " << i << " with element count: " << node_element_count << std::endl;
        //std::cout << "Min element id from vector: " << node_min_element_idx << std::endl;
        //std::cout << "Min element id from node: " << Node.min_elem_idx << std::endl;;
        //std::cout << "Contained elems: ";

        // Find strides for the given timestep to index correctly into Python buffers via pointers
        // Node coords are dimensioned as [timesteps, element count, nodes per element, coords per node]
        const int timestep_coords_stride = timestep * mesh_element_count * coords_per_element;
         // Face colors are dimensioned as [timesteps, element count, 2] IF using uvs_over_time (not default). Use this stride then
        //const int timestep_color_stride = timestep * mesh_element_count * uvs_per_element;

        // Iterate over elements in the node
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            // Get the index of the stored mesh element from the reshuffled vector of indices that was created in BLAS builder
            int original_element_idx = mesh_element_indices[element_idx];
            // Add element dimension stride to find min index of the nodes comprising current mesh element
            size_t original_element_idx_at_t = timestep_coords_stride + original_element_idx * coords_per_element; 

            //std::cout << "Original element idx in flat array: " << original_element_idx << " " << std::endl;

            // Find the corresponding index in the uvs array for the current element
            //size_t uv_idx_at_t = timestep_color_stride + original_element_idx * uvs_per_element; // If using uvs_over_time
            size_t uv_idx_at_t = original_element_idx * uvs_per_element; // UVs constant across all frames
            
            // Iterate over all nodes in this mesh element to copy the data
            for (int j = 0; j < Node.nodes_per_element; ++j){
                // Nodal coordinates
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES]); // x
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 1]); // y
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 2]); // z
                // Nodal normals
                Node.node_normals.push_back(mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES]); // x
                Node.node_normals.push_back(mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 1]); // y
                Node.node_normals.push_back(mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 2]); // z
                //std::cout << "Node normals: " << mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES] << " " << mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 1] << " " << mesh_node_normals_expanded_ptr[original_element_idx_at_t + j * NODE_COORDINATES + 2] << std::endl;
                // (u,v) for texturing
                Node.face_color.push_back(mesh_uvs_ptr[uv_idx_at_t + j * UV_COORDINATES]); // u
                Node.face_color.push_back(mesh_uvs_ptr[uv_idx_at_t + j * UV_COORDINATES + 1]); // v 
                //std::cout << "Face color size: " << Node.face_color.size() << std::endl;
                //Node.material.push_back(mesh_material);
                //Node.material.push_back(mesh_material);
                //Node.material.push_back(mesh_material);
            }
            
            //DEBUG VERSION. Does the same thing, but says very explicitly the indices, so they can be compared against a flat array in Python to see retrieved values etc.
            // Copy all uv coordinates for the mesh element
            /*
            std::cout << "\t UVs for this element: " << std::endl;
            std:: cout << "\t UV idx at t: " << uv_idx_at_t << std::endl;
            for (int j = 0; j < Node.nodes_per_element; ++j){
                std::cout << "\t Node number: " << j << std::endl;
                for (int k = 0; k < UV_COORDINATES; ++k){
                    std::cout << "\t\tUV number: " << k << " indexed as " << uv_idx_at_t + j * UV_COORDINATES + k << " with value: " << mesh_uvs_ptr[uv_idx_at_t + j * UV_COORDINATES + k] << std::endl;
                    Node.face_color.push_back(mesh_uvs_ptr[uv_idx_at_t + j * UV_COORDINATES + k]);
                }
            }
            std::cout << std::endl;
            */

            
        }
        //std::cout << "Node coords size: " << Node.node_coords.size() << std::endl;
       // std::cout << "Node element count: " << Node.element_count << std::endl;
    }
    //std::cout << "Total BVH coordinate count: " << coord_count << std::endl;
}

void copy_data_to_BLAS_node_color(BLAS &mesh_bvh,
    std::vector<int>& mesh_element_indices,
    std::vector<int>& node_minimum_element_index,
    const double* mesh_node_coords_expanded_ptr,
    const double* mesh_node_normals_expanded_ptr,
    const double* mesh_face_color_ptr,
    const int mesh_material,
    const int timestep){
    // Solid color version
    // Copies appropriate mesh data to store directly in BVH node, so it can be accessed easily upon intersection and be cache-friendly
    // This way we also avoid copying the mesh data when we move the node to the BVH tree vector as they're already there when we get to this part here.

    //std::cout << "BLAS builder: Copying mesh data into leaf nodes..." << std::endl;
    size_t bvh_node_count = mesh_bvh.tree_nodes.size();
    int mesh_element_count = mesh_element_indices.size();
   
    // Iterate over all BVH nodes
    for (int i = 0; i < bvh_node_count; ++i){
        BLAS_Node& Node = mesh_bvh.tree_nodes[i];

        // std::cout << Node.nodes_per_element << '\n';
        
        // Get indices of the mesh elements assigned to the node for the for loop 
        const int node_min_element_idx = node_minimum_element_index[i];
        const int node_element_count = Node.element_count;
        const int node_max_element_idx = node_min_element_idx + Node.element_count;
        const int coords_per_element = Node.nodes_per_element * NODE_COORDINATES; // number of nodes per element times 3 coordinates each
        Node.node_coords.reserve(node_element_count * coords_per_element);
        Node.node_normals.reserve(node_element_count * coords_per_element);
        Node.face_color.reserve(node_element_count * NODE_COORDINATES); // face_color will store 3 values
        //Node.material.reserve(node_element_count * NODE_COORDINATES);
        
        //std::cout << "BVH node id: " << i << " with element count: " << node_element_count << std::endl;
        //std::cout << "Min element id from vector: " << node_min_element_idx << std::endl;
        //std::cout << "Min element id from node: " << Node.min_elem_idx << std::endl;;
        //std::cout << "Contained elems: ";

        // Find strides for the given timestep to index correctly into Python buffers via pointers
        // Node coords are dimensioned as [timesteps, element count, nodes per element, coords per node]
        // Face colors are dimensioned as [timesteps, element count, coords per node]
        const int timestep_coords_stride = timestep * mesh_element_count * coords_per_element;
        const int timestep_color_stride = timestep * mesh_element_count * NODE_COORDINATES;

        // Iterate over elements in the node
        for (int element_idx = node_min_element_idx; element_idx < node_max_element_idx; ++element_idx){
            // Get the index of the stored mesh element from the reshuffled vector of indices that was created in BLAS builder
            int original_element_idx = mesh_element_indices[element_idx];
            // Add element dimension stride to find min index of nodes comprising current mesh element
            int original_element_idx_at_t = timestep_coords_stride + original_element_idx * coords_per_element; 

            //std::cout << "\nOriginal element id: " << original_element_idx << std::endl;
            //std::cout << "\tidx at t: " << original_element_idx_at_t << std::endl;
            
            // Copy all nodal coordinates
            //std::cout << "\t Node normals for this element: \n\t";
            for (int j = 0; j < coords_per_element; ++j){
                //std:: cout << mesh_node_coords_expanded_ptr[element_min_index + j] << " ";
                Node.node_coords.push_back(mesh_node_coords_expanded_ptr[original_element_idx_at_t + j]);
                Node.node_normals.push_back(mesh_node_normals_expanded_ptr[original_element_idx_at_t + j]);
                //std::cout << mesh_node_normals_expanded_ptr[original_element_idx_at_t + j] << " ";
            }
            // Copy all color (field) values for the mesh element
            size_t face_color_idx_at_t = timestep_color_stride + original_element_idx * NODE_COORDINATES;
            // Retrieve and copy 3 RGB values for this element to the BLAS node
            Node.face_color.push_back(mesh_face_color_ptr[face_color_idx_at_t]);
            Node.face_color.push_back(mesh_face_color_ptr[face_color_idx_at_t + 1]);
            Node.face_color.push_back(mesh_face_color_ptr[face_color_idx_at_t + 2]);

            //Node.material.push_back(mesh_material);
            //Node.material.push_back(mesh_material);
            //Node.material.push_back(mesh_material);
        }
        //std::cout << "Node coords size: " << Node.node_coords.size() << std::endl;
       // std::cout << "Node element count: " << Node.element_count << std::endl;
    }
    //std::cout << "Total BVH coordinate count: " << coord_count << std::endl;
}

void copy_data_to_TLAS(TLAS &tlas,
    std::vector<BLAS>& scene_BLASes,
    const std::vector<int>& scene_blas_indices){
    
    // Copy BLASes so they're stored in the traversal order determined by the builder, so data for each node is more local
    const size_t tlas_node_count = tlas.tlas_nodes.size();
    std::vector<BLAS>& blases_ordered = tlas.blases; 

    for(size_t i = 0; i < tlas_node_count; ++i){
        TLAS_Node& Node = tlas.tlas_nodes[i];
        // Iterate over all BLASes stored in TLAS nodes to copy them
        const int node_max_index = Node.min_blas_idx + Node.blas_count;
        for(int j = Node.min_blas_idx; j < node_max_index; ++j){
            int blas_idx = scene_blas_indices[j];
            blases_ordered.push_back(scene_BLASes[blas_idx]);
            blases_ordered[j].blas_idx = j;
        }
    }
 }

inline void set_BLAS_material(BLAS &mesh_bvh, const int mesh_material, const double mesh_ri, const double scene_ri, const enum ObjectType object_type){
    // Uncomment the below 2 lines if deciding to go for switch-based dispatcj in return_ray_color
    //mesh_bvh.material = mesh_material;
    //mesh_bvh.object_type = object_type;
    switch (mesh_material) {
        case UNLIT: {
            mesh_bvh.ray_material_ptr = &ray_unlit;
            mesh_bvh.refractive_index = scene_ri; // for non-refractive materials we expect mesh_ri = scene_ri, but just to be on the safe side
            return;
        }
        case DIFFUSE: { // Diffuse
            mesh_bvh.ray_material_ptr = &ray_diffuse;
            mesh_bvh.refractive_index = scene_ri;
            return;
    }
        case SPECULAR: {// Specular (mirror)
            mesh_bvh.ray_material_ptr = &ray_specular;
            mesh_bvh.refractive_index = scene_ri;
            return;
        }
        case REFRACTIVE: {// Refraction (dielectric)
            
            if (object_type == ObjectType::SOLID){
                mesh_bvh.ray_material_ptr = &ray_refractive<ObjectType::SOLID>;
            }
            else{
                mesh_bvh.ray_material_ptr = &ray_refractive<ObjectType::SHELL>;
            }
            mesh_bvh.refractive_index = mesh_ri;
            return;
        }
        default: { // Undefined - removed, so this should never, technically, get assigned
            mesh_bvh.ray_material_ptr = &ray_undefined;
            mesh_bvh.refractive_index = scene_ri;
            return;
        }
    }
}

// Unfortunately, these switches have to be this long if we want compile-time resolution of what happens inside these functions
inline void set_BLAS_intersection_texture(BLAS &mesh_bvh, const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type){
    // Assigns appropriate texture interpolation function pointer
    switch(nodes_per_element){
        case TRI3:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri3<ShadingType::FLAT, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::BLENDED: // For TRI3 blended and angle-averaged blended are the same thing, so it does not matter what gets picked here
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri3<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::TEXTURE>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case TRI6:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::FLAT, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::BLENDED, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::TEXTURE>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD4:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::FLAT, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::BLENDED, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::TEXTURE>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD8: 
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::FLAT, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::BLENDED, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::TEXTURE>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD9:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::FLAT, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::BLENDED, SurfaceType::TEXTURE>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::TEXTURE>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        default: throw std::invalid_argument("Unsupported element type.");
    }
}

inline void set_BLAS_intersection_color(BLAS &mesh_bvh,  const enum ElementNodeCount nodes_per_element, const enum ShadingType shading_type){
    // Assigns appropriate color interpolation function pointer
    switch(nodes_per_element){
        case TRI3:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri3<ShadingType::FLAT, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::BLENDED: // For TRI3 blended and angle-averaged blended are the same thing, so it does not matter what gets picked here
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri3<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case TRI6:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::FLAT, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_tri6<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD4:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::FLAT, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad4<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD8: 
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::FLAT, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad8<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        case QUAD9:
            switch(shading_type){
                case ShadingType::FLAT:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::FLAT, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                case ShadingType::ANGLE_AVG_BLENDED:
                    mesh_bvh.overwrite_intersection_function_ptr = &overwrite_intersection_quad9<ShadingType::ANGLE_AVG_BLENDED, SurfaceType::SOLID_COLOR>;
                    break;
                default: throw std::invalid_argument("Unsupported shading type.");
            }
            break;
            
        default: throw std::invalid_argument("Unsupported element type.");
    }
}


 // Helper/debug functions
 /*
inline void print_BLAS_data(BLAS& mesh_bvh){
    std::cout << "     BLAS has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;
    // Iterate over tree nodes
    for (int i = 0; i < mesh_bvh.tree_nodes.size(); ++i){
        std::cout << "          BLAS Node ID: " << i << std::endl;
        BLAS_Node& Node = mesh_bvh.tree_nodes[i];
        std::cout << "              Node coords vector size [elements]: " << Node.node_coords.size() << std::endl;
        std::cout << "              Node struct size total [bytes]: " << sizeof(Node) << std::endl;
        for (int j = 0; j < Node.element_count; ++j){
            std::cout << "              Mesh element ID: " << j << std::endl;
            int element_base_idx = j * Node.nodes_per_element * NODE_COORDINATES;

            for (int k = 0; k < Node.nodes_per_element; ++k){
                int node_base_idx = element_base_idx + k * NODE_COORDINATES;
                std::cout << "                Node: " << k << std::endl;      
                std::cout << "                  Node normal: ";
                for (int z = 0; z < NODE_COORDINATES; ++z){
                    std::cout << Node.node_normals[node_base_idx + z] << " ";
                }
                std::cout << std::endl;
                std::cout << "                  Node coords: ";   
                for (int z = 0; z < NODE_COORDINATES; ++z){
                    std::cout << Node.node_coords[node_base_idx + z] << " ";
                }
            std::cout << std::endl;
        }
    }
}
}

inline void print_TLAS(TLAS &scene_TLAS){
    for (int i = 0; i < scene_TLAS.tlas_nodes.size(); ++i){
        std::cout << "TLAS Node ID: " << i << std::endl;
        TLAS_Node& Node = scene_TLAS.tlas_nodes[i];
        std::cout << "  Node BLAS count: " << Node.blas_count << std::endl;
        std::cout << "  Node min index: " << Node.min_blas_idx << std::endl;
        std::cout << "  Printing contained BLASes..." << std::endl;
        for(int j = Node.min_blas_idx; j < Node.blas_count; ++j){
            BLAS& mesh_bvh = scene_TLAS.blases[j];
            print_BLAS_data(mesh_bvh);
        }
    }
 }
*/

// Main function allowing mixed surface types
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
    const int timestep_count){
// Handles building all acceleration structures in the scene - bottom and top level

    size_t scene_mesh_count = scene_coords_expanded.size(); 
   
    // All containers to store the data in the scene
    std::vector<std::array<double,3>> scene_blas_centroids; // Stores centroids of the whole objectes (meshes) in this scene
    scene_blas_centroids.reserve(scene_mesh_count);
    std::vector<AABB> scene_blas_aabbs; // Store AABBs of the whole objects in this scene
    scene_blas_aabbs.reserve(scene_mesh_count);
    std::vector<BLAS> scene_blases; // Store mesh_bvhs - this will be used for TLAS
    scene_blases.reserve(scene_mesh_count);

    // Get the refractive index of the scene (typically air, but in case it is not)
    const int last_index = scene_refractive_indices.size() - 1;
    const float scene_ri = scene_refractive_indices[last_index]; // Scene RI is stored at the last position always

    // Get shading type
    ShadingType shading_type_enum = static_cast<ShadingType>(shading_type);

    // Iterate over MESHES to build BLASes - BVHs for respective meshes
    for (size_t mesh_idx = 0; mesh_idx < scene_mesh_count; ++mesh_idx) {
        
        // 1. Geometric part dependent on the surface element type

        // Access data from Python buffer for this particular mesh (i.e., scene->object)
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_coords = scene_coords_expanded[mesh_idx];
        enum ElementNodeCount nodes_per_element = ElementNodeCount(mesh_node_coords.shape(2));
        // size_t mesh_element_count = mesh_node_coords.shape(0); // number of elements comprising the mesh WITHOUT timesteps
        size_t mesh_element_count = mesh_node_coords.shape(1); // number of elements comprising the mesh WITH TIMESTEPS

        std::cout << "Mesh: " << mesh_idx << "; Timesteps: " << mesh_node_coords.shape(0) << 
                                              "; Elements: " << mesh_node_coords.shape(1) << 
                                     "; Nodes per element: " << mesh_node_coords.shape(2) <<
                                  "; Coordinates per node: " << mesh_node_coords.shape(3) << '\n';

        // Containers for calculated data for this mesh
        std::vector<std::array<double, NODE_COORDINATES>> mesh_element_centroids; // Store centroids for this mesh
        mesh_element_centroids.reserve(mesh_element_count);
        std::vector<AABB> mesh_element_aabbs; // Bounding volumes for the elements in this mesh
        mesh_element_aabbs.reserve(mesh_element_count);
        scene_blas_aabbs.emplace_back();
        AABB& mesh_aabb = scene_blas_aabbs[mesh_idx]; // AABB for the entire mesh

        // Pointer to access data for copying into BVH nodes - much faster than doing it through nanobind interface. Lifetime managed by Python
        double* mesh_node_coords_ptr = const_cast<double*>(mesh_node_coords.data());
        
        // Create BLAS (BVH) for this mesh
        //std::cout << "Generating BLAS for mesh " << mesh_idx << std::endl;
        scene_blases.emplace_back(); // Generate directly inside the vector to avoid copying data
        BLAS& mesh_bvh = scene_blases[mesh_idx]; // Get a reference to the BVH of the current mesh to pass it to the builder functions

        // Iterate over ELEMENTS in this mesh (types specified in enum in rtelemconstants.h) 
        // And assign appropriate intersection function pointers based on the element type here, so we do not need to run these checks in intersection hot loops
        switch(nodes_per_element){
            case TRI3:
                process_element_data<TRI3>(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
                mesh_bvh.intersection_function_ptr = &intersect_bvh_tri3;
                break;
            case TRI6:
                process_element_data<TRI6>(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
                mesh_bvh.intersection_function_ptr = &intersect_bvh_tri6;
                break;
            case QUAD4:
                process_element_data<QUAD4>(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
                mesh_bvh.intersection_function_ptr = &intersect_bvh_quad4;
                break;
            case QUAD8:
                process_element_data<QUAD8>(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
                mesh_bvh.intersection_function_ptr = &intersect_bvh_quad8;
                break;
            case QUAD9:
                process_element_data<QUAD9>(mesh_element_count, mesh_node_coords_ptr, mesh_element_centroids, mesh_element_aabbs, mesh_aabb, timestep);
                mesh_bvh.intersection_function_ptr = &intersect_bvh_quad9;
                break;
            default: throw std::invalid_argument("Unsupported element type."); // Shouldn't ever get triggered since we check element type on the Python side as well
        }
     
        // Find centroid of the entire mesh
        scene_blas_centroids.emplace_back();
        std::array<double,NODE_COORDINATES>& mesh_centroid = scene_blas_centroids[mesh_idx];
        compute_mesh_centroid(mesh_aabb, mesh_centroid);

        // Temporary vectors to reshuffle element indices as we build the BVH, then use this mapping
        // to append the mesh data in the nodes instead of needing to access it at the split time
        std::vector<int> mesh_element_indices;
        mesh_element_indices.resize(mesh_element_count);
        std::iota(mesh_element_indices.begin(), mesh_element_indices.end(), 0);
        std::vector<int> node_minimum_element_index; // Instead of wasting BLAS_Node struct space on storing this value


        // 2. BLAS BVH builder functions - this part depends on the surface type
        build_BLAS(mesh_bvh, mesh_element_centroids, mesh_element_aabbs, mesh_element_indices, node_minimum_element_index, mesh_element_count, nodes_per_element);
        int mesh_material = materials[mesh_idx];
        double mesh_ri = scene_refractive_indices[mesh_idx];
        ObjectType mesh_object_type = static_cast<ObjectType>(mesh_object_types[mesh_idx]);
        set_BLAS_material(mesh_bvh, mesh_material, mesh_ri, scene_ri, mesh_object_type);
        mesh_bvh.priority = mesh_priorities[mesh_idx];
        mesh_bvh.thickness = scene_mesh_thickness[mesh_idx];
       
        int surface_type = scene_surface_types[mesh_idx];
		nanobind::ndarray<const double, nanobind::c_contig> mesh_node_normals = scene_normals_expanded[mesh_idx];
        double* mesh_node_normals_ptr = const_cast<double*>(mesh_node_normals.data()); // Index into the node normals to copy them

        // Uncomment if in doubt the data from Python gets passed correctly before it is copied
        /*
        for (int element = 0; element < 10; ++element){
            std::cout << "\nElement: " << element << std::endl;
            for (int node = 0; node < 3; ++node){
                int base_idx = element * 3 * NODE_COORDINATES;
                int i = base_idx + node * NODE_COORDINATES;
                std::cout << "\n\t Node " << node << ": ";
                std::cout << "\n\t\tCoords: " << mesh_node_coords_ptr[i] << " " << mesh_node_coords_ptr[i+1] << " " << mesh_node_coords_ptr[i+2] << " ";
                std::cout << "\n\t\tNormal: " <<  mesh_node_normals_ptr[i] << " " << mesh_node_normals_ptr[i+1] << " " << mesh_node_normals_ptr[i+2] << " ";
            }
        }
        std::cout << std::endl;*/

        if (surface_type == 1){ // Texture
            nanobind::ndarray<const double, nanobind::c_contig> mesh_texture_arr = scene_textures[mesh_idx];
            double* mesh_texture_ptr = const_cast<double*>(mesh_texture_arr.data());
            Texture mesh_texture(mesh_texture_ptr, mesh_texture_arr.shape(0), mesh_texture_arr.shape(1)); // Pointer, height, width
            mesh_bvh.texture = mesh_texture; // Assign texture struct to the BLAS
            nanobind::ndarray<const double, nanobind::c_contig> mesh_uvs = scene_uvs[mesh_idx];
            double* mesh_uvs_ptr = const_cast<double*>(mesh_uvs.data());
            // DEBUG to ensure data is copied correctly
            /*
            std::cout << "Mesh uvs shape for this mesh as extracted from scene: " << std::endl;
            std::cout << mesh_uvs.shape(0) << " " << mesh_uvs.shape(1) << " " << mesh_uvs.shape(2) << " " << mesh_uvs.shape(3) << std::endl;
            for (int i = 0; i < 264; ++i){
                std::cout << mesh_uvs_ptr[i] << " ";
            }
            std::cout << std::endl;
            */

            copy_data_to_BLAS_node_tex(mesh_bvh, mesh_element_indices, node_minimum_element_index, mesh_node_coords_ptr, mesh_node_normals_ptr, mesh_uvs_ptr, mesh_material, timestep);
            set_BLAS_intersection_texture(mesh_bvh, nodes_per_element, shading_type_enum);
           
        }
        else if (surface_type == 0){ // Solid surface fill
            nanobind::ndarray<const double, nanobind::c_contig> mesh_face_colors = scene_face_colors[mesh_idx];
            double* mesh_face_colors_ptr = const_cast<double*>(mesh_face_colors.data());
            copy_data_to_BLAS_node_color(mesh_bvh, mesh_element_indices, node_minimum_element_index, mesh_node_coords_ptr, mesh_node_normals_ptr, mesh_face_colors_ptr, mesh_material, timestep);
            set_BLAS_intersection_color(mesh_bvh, nodes_per_element, shading_type_enum);
        }
        else {
            throw std::invalid_argument("Unsupported surface type."); // Shouldn't ever get triggered since we check element type on the Python side as well, but might be useful for debugging
        }


        //std::cout << "BLAS successfully built." << std::endl;
        //std::cout << "BVH has " << mesh_bvh.tree_nodes.size() << " nodes." << std::endl;
        //print_BLAS_data(mesh_bvh);

    } //MESHES

    // BUILD TLAS - structure of BLASes
    TLAS scene_TLAS;
    scene_TLAS.tlas_nodes.reserve(scene_mesh_count);
    scene_TLAS.blases.reserve(scene_blases.size()); // Can guarantee this size as it will store all BLASes, just re-shuffled
    // Temporary vector to reshuffle element indices as we build the BVH, instead of having to access the mesh data all the time to append it in nodes right away as we do so
    
    // TLAS is much smaller, so in this case we will be keeping the vector with indices and using it to index into BLASes stored in the node
    std::vector<int> scene_blas_indices;
    scene_blas_indices.resize(scene_mesh_count);
    std::iota(scene_blas_indices.begin(), scene_blas_indices.end(), 0);

    // TLAS BVH builder functions
    build_TLAS(scene_TLAS.tlas_nodes, scene_blas_centroids, scene_blas_aabbs, scene_blas_indices, scene_mesh_count);
    copy_data_to_TLAS(scene_TLAS, scene_blases, scene_blas_indices);
    //std::cout << "TLAS successfully built." << std::endl;
    //Ray test_ray;
    //test_ray.origin = EiVector3d(0.0, 0.0, 0.0);
    //test_ray.direction = EiVector3d(1.0, 0.0, 0.0);
    //intersect_tlas(test_ray, scene_TLAS);
    //print_TLAS(scene_TLAS);

    return scene_TLAS;
 } // SCENE (end of function)