// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD header files
#include <iostream>
#include <limits>

// ray tracer header files
#include "rtrayintersection.h"

inline EiVector3d get_face_color(Eigen::Index minRowIndex,
    const std::vector<double>& face_color) {
    // Get values to colour the intersected face
    double c1 = face_color[minRowIndex * 3];
    double c2 = face_color[minRowIndex * 3 + 1];
    double c3 = face_color[minRowIndex * 3 + 2];
    EiVector3d face_color_vec;
    //face_color_vec << 0.0, 0.0, 0.0;
    face_color_vec << c1, c2, c3;
    return face_color_vec;
}

EiVectorD3d cross_rowwise(const EiVectorD3d& mat1, const EiVectorD3d& mat2) {
    // Row-wise cross product for 2 matrices (i.e., treating each row as a vector).
    // Also works for multiplying a matrix with a row vector, so the input order determines the multiplication order. Happy days.
    // Written because this otherwise can't be a one-liner like in NumPy - Eigen's cross product works only for vector types.
    if (mat1.cols() != 3 || mat2.cols() != 3) {
        std::cerr << "Error: matrices need to have exactly 3 columns to find the cross product" << std::endl;
        return {};
    }
    long long number_of_rows = mat1.rows(); // number of rows. Long long to match the type from Eigen::Index
    EiVectorD3d cross_product_result(number_of_rows, 3);
    cross_product_result.col(0) = mat1.col(1).cwiseProduct(mat2.col(2)) - mat1.col(2).cwiseProduct(mat2.col(1));
    cross_product_result.col(1) = mat1.col(2).cwiseProduct(mat2.col(0)) - mat1.col(0).cwiseProduct(mat2.col(2));
    cross_product_result.col(2) = mat1.col(0).cwiseProduct(mat2.col(1)) - mat1.col(1).cwiseProduct(mat2.col(0));
    return cross_product_result;
}

IntersectionOutput intersect_bvh_triangles(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_triangle_count) {

    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(bvh_node_triangle_count, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(bvh_node_triangle_count, 1).array();

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_triangle_count, 3),
        EiVectorD3d::Zero(bvh_node_triangle_count, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_triangle_count, 1, std::numeric_limits<double>::infinity())
    };

    // Calculations - edges and normals
    EiMatrixDd edge0(bvh_node_triangle_count, 3), nEdge2(bvh_node_triangle_count, 3); // shape (faces, 3) each
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  nodes0(bvh_node_triangle_count, 3);
    for (int triangle_idx = 0; triangle_idx < bvh_node_triangle_count; triangle_idx++) {
        int node_0 = triangle_idx * 3;
        int node_1 = triangle_idx * 3 + 1;
        int node_2 = triangle_idx * 3 + 2;
        //std::cout << "Triangle node indices: " << node_0 << " " << node_1 << " " << node_2 << std::endl;

        for (int j = 0; j < 3; j++) {
            //std::cout<<node_coords_arr[i][j] << " ";
            //edge0(i, j) = node_coords.at(node_1, j) - node_coords.at(node_0, j);
            edge0(triangle_idx, j) = node_coords[node_1 * 3 + j] - node_coords[node_0 * 3 + j];
            //std::cout << "node_coords at " << node_1 *3 + j << " are: " << node_coords[node_1 * 3 + j] << std::endl;
            //std::cout << "edge 0: " << edge0(triangle_idx,j) << std::endl;
            //nodes0(i, j) = node_coords.at(node_0, j);
            nodes0(triangle_idx, j) = node_coords[node_0 * 3 + j];
            //std::cout << "nodes0 : " << nodes0(triangle_idx,j) << std::endl;
            // Skip edge1 because it never gets used in the calculations anyway
            //nEdge2(i, j) = node_coords.at(node_2, j) - node_coords.at(node_0, j);
            nEdge2(triangle_idx, j) = node_coords[node_2 * 3 + j] - node_coords[node_0 * 3 + j];
            //std::cout << "nEdge2 : " << nEdge2(triangle_idx,j) << std::endl;
        }
    }
    EiVectorD3d plane_normals = cross_rowwise(edge0, nEdge2); // not normalised! Shape (faces, 3)

    // Step 1: Quantities for the Moller Trumbore method
    EiArrayD3d p_vec = cross_rowwise(ray_directions, nEdge2); // Assigns a vector to an array variable, but Eigen automatically converts so long as the underlying sizes are correct at initialization. Shape (faces, 3)
    Eigen::Array<double, Eigen::Dynamic, 1> determinants = (edge0.array() * p_vec).rowwise().sum(); // Row-wise dot product; shape (faces, 1)

    // Step 2: Culling.
    //Determinant negative -> triangle is back-facing. If det is close to 0, ray and triangle are parallel and ray misses the triangle.
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask = (determinants > 1e-6) && (determinants > 0);
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    // Step 3: Test if ray is in front of the triangle
    Eigen::Array<double, Eigen::Dynamic, 1> inverse_determinants = determinants.inverse(); // Element-wise inverse. shape (faces, 1)
    EiArrayD3d t_vec = ray_origins - nodes0; // shape (faces, 3)
    Eigen::Array<double, Eigen::Dynamic, 1> barycentric_u = ((t_vec * p_vec).rowwise().sum()).array() * inverse_determinants; // shape (faces, 1)
    valid_mask = valid_mask && (barycentric_u >= 0) && (barycentric_u <= 1);
    if (!valid_mask.any()) {
        //std::cout << "Condition 2 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

    EiArrayD3d q_vec = cross_rowwise(t_vec.matrix(), edge0); // shape (faces, 3)
    Eigen::Array<double, Eigen::Dynamic, 1> barycentric_v = (ray_directions.array() * q_vec).rowwise().sum().matrix().array() * inverse_determinants; // shape (faces, 1)
    // Check barycentric_v and sum
    valid_mask = valid_mask && (barycentric_v >= 0) && ((barycentric_u + barycentric_v) <= 1);
    // t values
    Eigen::Array<double, Eigen::Dynamic, 1> t_values = (nEdge2.array() * q_vec).rowwise().sum().array() * inverse_determinants; // shape (faces, 1)
    valid_mask = valid_mask && (t_values >= ray.t_min) && (t_values <= ray.t_max);

    // Iterate through all t_values and set them to infinity if they don't satisfy the conditions imposed by the mask
    for (int i = 0; i < t_values.rows(); ++i) {
        for (int j = 0; j < t_values.cols(); ++j) {
            if (!valid_mask(i, j)) {
                t_values(i, j) = std::numeric_limits<double>::infinity();
            }
        }
    }
    // Create an array for barycentric coordinates so we can do things element-wise with those
    Eigen::ArrayXXd barycentric_coordinates(bvh_node_triangle_count, 3);
    barycentric_coordinates.col(0) = barycentric_u;
    barycentric_coordinates.col(1) = barycentric_v;
    barycentric_coordinates.col(2) = 1.0 - barycentric_u - barycentric_v; // barycentric_w
    return IntersectionOutput{ barycentric_coordinates, plane_normals, t_values };
}

bool intersect_AABB (const Ray& ray, const AABB& AABB) {
    // Slab method for ray-AABB intersection
    double t_axis[6]; // t values for each axis, so [0,1] are for x, [2,3] for y, and [4,5] for z
    EiVector3d inverse_direction = 1/(ray.direction.array()); // Divide first to use cheaper multiplication later

    // Find ray intersections with planes defining the AABB in X, Y, Z
    for (int i = 0; i < 3; ++i) {
        t_axis[2*i] = (AABB.corner_min[i] - ray.origin(i)) * inverse_direction(i);
        t_axis[2*i + 1] = (AABB.corner_max[i] - ray.origin(i)) * inverse_direction(i);
    }
    //Overlap test
    // Find the minimum t for each axis (x, y, z), then find maximum of these for (x,y,z)
    double t_min = std::max(std::max(std::min(t_axis[0], t_axis[1]), std::min(t_axis[2], t_axis[3])), std::min(t_axis[4], t_axis[5]));
    // Find the maximum t for each axis (x, y, z), then find minimum of these for (x,y,z)
    double t_max = std::min(std::min(std::max(t_axis[0], t_axis[1]), std::max(t_axis[2], t_axis[3])), std::max(t_axis[4], t_axis[5]));

    // t_min < t_max - Ray which just touches a corner, edge, or face of the AABB will be considered non-intersecting
    // t_min <= t_max - Rays which touch the box boundary are considered intersecting. A bit of a degenerate case, but decided to include it here, hence more relaxed inequality.
    // t_min < ray.t_max - Clip to ray segment
    return t_min <= t_max && t_max > 0.0 && t_min < ray.t_max; // False => No overlap => Ray does not intersect the AABB.
}

void intersect_BLAS(const Ray& ray,
    const BLAS& mesh_bvh,
    IntersectionOutput &out_intersection,
    HitRecord &intersection_record) {

     //std::cout << "  BLAS: Starting BVH intersection test" << std::endl;
     //const BLAS_Node& root = mesh_bvh.tree_nodes[mesh_bvh.root_idx];

     std::vector<int> stack; // Store node indices on the stack
     stack.push_back(mesh_bvh.root_idx);

     while(!stack.empty()){
        const BLAS_Node& Node = mesh_bvh.tree_nodes[stack.back()];
        stack.pop_back();

        // Debug notes: Renders wrong if I uncomment below. But renders ok if I don't
        // So all triangle data per node is still preserved, which is good
        // => intersect AABB is wrong? calculating AABB? Like this suggests that we exit prematurely
        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB of the node

        if (Node.left_child_idx == -1) {
            // No children => Leaf node => Intersect triangles
           
            out_intersection = intersect_bvh_triangles(ray, Node.node_coords, Node.element_count);
            Eigen::Index minRowIndex, minColIndex;
            //std::cout << "Number of t_values: " << out_intersection.t_values.size() << std::endl;

            out_intersection.t_values.minCoeff(&minRowIndex, &minColIndex); // Find indices of the smallest t_value
            double closest_t = out_intersection.t_values(minRowIndex, minColIndex);
            //std::cout << "Closest t found: " << closest_t << std::endl;

            if (closest_t < intersection_record.t) {
                intersection_record.t = closest_t;
                intersection_record.barycentric_coordinates = out_intersection.barycentric_coordinates.row(minRowIndex);
                intersection_record.point_intersection = ray_at_t(closest_t, ray);
                intersection_record.normal_surface = out_intersection.plane_normals.row(minRowIndex);
                intersection_record.face_color = get_face_color(minRowIndex, Node.face_color);
            }
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
            // Potential improvement: testing node distance vs. ray to push the farther one first, so we trasverse closer child first.
            // How to: Compare t_near from intersect_AABB for both children and intersect the closer one first
        }   
     }
}

void intersect_TLAS(const Ray& ray,
    const TLAS& scene_TLAS,
    IntersectionOutput &out_intersection,
    HitRecord &out_intersection_record){

    //std::cout << "TLAS: Starting BVH intersection test" << std::endl;
     std::vector<int> stack; // Store node indices on the stack
     stack.push_back(0); // Push root index

     while(!stack.empty()){
        const TLAS_Node& Node = scene_TLAS.tlas_nodes[stack.back()];
        stack.pop_back();

        if (!intersect_AABB(ray, Node.bounding_box)) continue; // Early exit if ray does not intersect the AABB of the node
        if (Node.left_child_idx == -1) {
            // No children => Leaf node => Intersect triangles
            //std::cout << "TLAS: Leaf node reached with " << Node.blas_count << " BLASes." << std::endl;
            int node_max_index = Node.min_blas_idx + Node.blas_count;
            for (int i = Node.min_blas_idx; i < node_max_index; ++i){
                //std::cout << " TLAS: Intersected BLAS index: " << i << std::endl;
                intersect_BLAS(ray, scene_TLAS.blases[i], out_intersection, out_intersection_record);
            }
        }
        else { // Not a leaf node => Test children nodes for intersections
            // DFS order
            int left = Node.left_child_idx;
            int right = left + 1;
            if (right != 0) stack.push_back(right);
            if(left != -1) stack.push_back(left);
        }
     }
}