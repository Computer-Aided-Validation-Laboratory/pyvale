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

constexpr int NODE_COORDINATES = 3; // number of coordinates per each mesh node. Used for some of flat indexing

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

    // Go through all the triangles and find an intersection of each triangle with a ray

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

IntersectionOutput intersect_bvh_quads(const Ray& ray,
    const std::vector<double>& node_coords,
    const unsigned int bvh_node_quad_count){
    // Go through all the quads and find an intersection of each quad with a ray
    
    // Ray data broadcasted to use in vectorised operations on matrices
    // This is faster than doing it in a loop
    EiVectorD3d ray_directions = ray.direction.replicate(bvh_node_quad_count, 1);
    EiArrayD3d ray_origins = ray.origin.replicate(bvh_node_quad_count, 1).array();

    // Define default negative output if there is no intersection
    static IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_count, 3),
        EiVectorD3d::Zero(bvh_node_quad_count, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_count, 1, std::numeric_limits<double>::infinity())
    };

    // Calculations - edges and normals
    EiMatrixDd edge0(bvh_node_quad_count, 3), nEdge2(bvh_node_quad_count, 3); // shape (faces, 3) each
    

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>  nodes0(bvh_node_quad_count, 3);

    for (int quad_idx = 0; quad_idx < bvh_node_quad_count; quad_idx++) {
        int node_0 = quad_idx * 3;
        int node_1 = quad_idx * 3 + 1;
        int node_2 = quad_idx* 3 + 2;
        //std::cout << "Quad node indices: " << node_0 << " " << node_1 << " " << node_2 << std::endl;

        for (int j = 0; j < 3; j++) {
            //std::cout<<node_coords_arr[i][j] << " ";
            //edge0(i, j) = node_coords.at(node_1, j) - node_coords.at(node_0, j);
            edge0(quad_idx, j) = node_coords[node_1 * 3 + j] - node_coords[node_0 * 3 + j];
            //std::cout << "node_coords at " << node_1 *3 + j << " are: " << node_coords[node_1 * 3 + j] << std::endl;
            //std::cout << "edge 0: " << edge0(triangle_idx,j) << std::endl;
            //nodes0(i, j) = node_coords.at(node_0, j);
            nodes0(quad_idx, j) = node_coords[node_0 * 3 + j];
            //std::cout << "nodes0 : " << nodes0(triangle_idx,j) << std::endl;
            // Skip edge1 because it never gets used in the calculations anyway
            //nEdge2(i, j) = node_coords.at(node_2, j) - node_coords.at(node_0, j);
            nEdge2(quad_idx, j) = node_coords[node_2 * 3 + j] - node_coords[node_0 * 3 + j];
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

    }

struct Ray_old { Eigen::Vector3d o, d; Ray_old(Eigen::Vector3d o_, Eigen::Vector3d d_) : o(o_), d(d_) {} };

void ray_convert(Ray_old &ray_old, const Ray &ray) {

    for (int i = 0; i < 3; ++i) {
        ray_old.o(i) = ray.origin(i);
        ray_old.d(i) = ray.direction(i);
    }
}

struct Quadratic_tet {
    std::vector<Eigen::Vector3d> nodes;

    // Nodes for the 4 faces - each face has 6 nodes: 3 corners, 3 mid-edges
    // Numbering for 1 tetrahedron example
    // const int face_indices[4][6] = {
    //     {0, 2, 1, 6, 5, 4}, // Face 0 (bottom)
    //     {0, 1, 3, 4, 8, 7}, // Face 1
    //     {1, 2, 3, 5, 9, 8}, // Face 2
    //     {2, 0, 3, 6, 7, 9}  // Face 3
    // };

      // Correct numbering for NGSolve/Netgen
      const int face_indices[4][6] = {
      {0, 1, 2, 4, 7, 5}, // Face 0 (bottom) (v0, v1, v2)
      {0, 1, 3, 4, 8, 6}, // Face 1 (v0, v1, v3)
      {1, 2, 3, 7, 9, 8}, // Face 2 (v1, v2, v3)
      {0, 2, 3, 5, 9, 6}  // Face 3 (v0, v2, v3)
      };

    Quadratic_tet(std::vector<Eigen::Vector3d> nodes_) :
        nodes(nodes_) {}

    // Quadratic triangle shape functions (g, h)
    // r = 1 - g - h
    static Eigen::VectorXd get_face_N(double g, double h) {
        double r = 1.0 - g - h;
        Eigen::VectorXd N(6);
        N << r*(2*r-1), g*(2*g-1), h*(2*h-1), 4*g*r, 4*g*h, 4*h*r;
        return N;
    }

    static Eigen::Matrix<double, 3, 2> get_face_Jacobian(double g, double h, const std::vector<Eigen::Vector3d>& f_nodes) {
        double r = 1.0 - g - h;
        // Derivatives of N wrt g and h
        double dNdu[6] = { -(4*r-1), 4*g-1, 0, 4*(r-g), 4*h, -4*h };
        double dNdv[6] = { -(4*r-1), 0, 4*h-1, -4*g, 4*g, 4*(r-h) };

        Eigen::Matrix<double, 3, 2> J = Eigen::Matrix<double, 3, 2>::Zero();
        for (int i = 0; i < 6; ++i) {
            J.col(0) += f_nodes[i] * dNdu[i];
            J.col(1) += f_nodes[i] * dNdv[i];
        }
        return J;
    }

    double intersect(const Ray_old &r, Eigen::Vector3d &n_out, Eigen::Vector2d &uv) const {
      
    // Define imprecision parameters for the initial guess, they are especially needed for the cases
    // where the ray is close to being tangential to the linear triangle.
    // If the determinant is close to zero, the ray misses the triangle.
    // Imprecision levels should be relatively large for the initial guess.
    const double eps_init_guess1 = 1e-10; // Imprecision for linear tringle's determinant
    const double eps_init_guess2 = 0.1; // Imprecision for linear tringle's isoparametric coordinates (u, v)
    const double eps_t = 1e-5; // Imprecision for t

    const int iter_max = 50; // Maximum number of iterations for Newton-Raphson method.

      // Define imprecision parameters for the solution.
      // Imprecision levels should be relatively small for solution.
      const double eps_sol1 = 1e-7; // Imprecision for the residual
      const double eps_sol2 = 1e-8; // Imprecision for the quadratic triangle's isoparametric coordinates (g, h)
      const double eps_sol3 = 1e-10; // Imprecision for the determinant

    double min_t = std::numeric_limits<double>::infinity();
    bool intersect = false;

    // 1. Intersect ray with the 4 triangles
    // Use the code already used for Traingle structure.
    // Impose the defined imprecision levels

    // Mapping for sub-triangulation (indices within the 6-node f_nodes vector)
    // Quadratic layout: 0,1,2 are corners; 3,4,5 are midpoints of (0-1), (1-2), (2-0)
    int sub_tris[4][3] = {
        {0, 3, 5}, // Bottom-left
        {3, 1, 4}, // Bottom-right
        {5, 4, 2}, // Top
        {3, 4, 5}  // Center
    };

    // Barycentric coordinate offsets for the sub-triangles to map back to (g, h)
    // These represent the (g, h) coordinates of the nodes in f_nodes
    Eigen::Vector2d nodes_gh[6] = {
        {0,0}, {1,0}, {0,1}, {0.5,0}, {0.5,0.5}, {0,0.5}
    };

    for (int f = 0; f < 4; ++f) {
        std::vector<Eigen::Vector3d> f_nodes(6);
        for(int i=0; i<6; ++i) f_nodes[i] = nodes[face_indices[f][i]];

        // Sub-triangulation
        double best_sub_t = std::numeric_limits<double>::infinity();
        Eigen::Vector2d best_gh_guess(0.33, 0.33);
        bool found_guess = false;

        for (int s = 0; s < 4; ++s) {
            Eigen::Vector3d v0 = f_nodes[sub_tris[s][0]];
            Eigen::Vector3d v1 = f_nodes[sub_tris[s][1]];
            Eigen::Vector3d v2 = f_nodes[sub_tris[s][2]];

            Eigen::Vector3d edge1 = v1 - v0, edge2 = v2 - v0;
            Eigen::Vector3d pvec = r.d.cross(edge2);
            double det = edge1.dot(pvec);
            if (fabs(det) < eps_init_guess1) continue;

            double invDet = 1.0 / det;
            Eigen::Vector3d tvec = r.o - v0;
            double u_sub = tvec.dot(pvec) * invDet;
            if (u_sub < 0 - eps_init_guess2 || u_sub > 1 + eps_init_guess2) continue;

            Eigen::Vector3d qvec = tvec.cross(edge1);
            double v_sub = r.d.dot(qvec) * invDet;
            if (v_sub < 0 - eps_init_guess2 || (u_sub + v_sub) > 1 + eps_init_guess2) continue;

            double t_sub = edge2.dot(qvec) * invDet;
            if (t_sub > eps_t && t_sub < best_sub_t) {
                best_sub_t = t_sub;
                // Interpolate the global (g, h) from the sub-triangle's local barycentrics
                double w_sub = 1.0 - u_sub - v_sub;
                best_gh_guess = w_sub * nodes_gh[sub_tris[s][0]] + 
                                u_sub * nodes_gh[sub_tris[s][1]] + 
                                v_sub * nodes_gh[sub_tris[s][2]];
                found_guess = true;
            }
        }

        if (!found_guess) continue;

        // 2. Start Newton-Raphson at t_guess, and gh from the right triangle's centroid 
        // or approximated (u, v) from the linear triangle. 
        Eigen::Vector2d gh = best_gh_guess;
        double t = best_sub_t;

        for (int iter = 0; iter < iter_max; ++iter) {
            Eigen::VectorXd N = get_face_N(gh.x(), gh.y());
            Eigen::Vector3d P = Eigen::Vector3d::Zero();
            for(int i=0; i<6; ++i) P += N[i] * f_nodes[i];

            Eigen::Vector3d res = r.o + t * r.d - P;
            if (res.norm() < eps_sol1) {
                // Check if hit is within quadratic triangle bounds
                    if (gh.x() >= 0 - eps_sol2 && gh.y() >= 0 - eps_sol2 && (gh.x() + gh.y()) <= 1 + eps_sol2) {
                    if (t < min_t && t > eps_t) {
                        min_t = t;
                        Eigen::Matrix<double, 3, 2> J = get_face_Jacobian(gh.x(), gh.y(), f_nodes);
                        Eigen::Vector3d normal = J.col(0).cross(J.col(1)).normalized();

                            // Ensure normal points against the ray (outward for convex)
                            if (normal.dot(r.d) > 0) {
                                normal = -normal;
                            }
                            n_out = normal;
                            uv = gh;
                            intersect = true;
                    }
                }
                break;
            }

            // Solve [rd, -dP/du, -dP/dv] * [dt, du, dv]^T = -res
            Eigen::Matrix<double, 3, 2> J = get_face_Jacobian(gh.x(), gh.y(), f_nodes);
            Eigen::Matrix3d M;
            M.col(0) = r.d;
            M.col(1) = -J.col(0);
            M.col(2) = -J.col(1);

            if (std::abs(M.determinant()) < eps_sol3) break;
            
            // Eigen::Vector3d delta = M.colPivHouseholderQr().solve(-res);
            Eigen::Vector3d delta = M.inverse() * (-res);
            t += delta.x();
            gh.x() += delta.y();
            gh.y() += delta.z();

            if (gh.x() < -0.5 || gh.y() < -0.5 || (gh.x() + gh.y()) > 1.5) break;
        }
    }

    return intersect ? min_t : std::numeric_limits<double>::infinity();
}
};



void load_quad_tets(const std::vector<double>& node_coords, 
                    std::vector<Quadratic_tet> &quadratic_tets,
                    const unsigned int bvh_node_quad_tet_count,
                    enum ElementNodeCount nodes_per_element) {
                    

   const int coords_per_element = nodes_per_element * NODE_COORDINATES; // number of elements times 3 coordinates each

   for (int quad_tet_idx = 0; quad_tet_idx < bvh_node_quad_tet_count; quad_tet_idx++) {

    std::vector<Eigen::Vector3d> nodes;
    
    int index_min = quad_tet_idx * coords_per_element;

        for (int i = 0; i < 10; i++) {

            Eigen::Vector3d node(0, 0, 0);

            node(0) = node_coords[index_min + i * 3 + 0]; // X-component
            node(1) = node_coords[index_min + i * 3 + 1]; // Y-component
            node(2) = node_coords[index_min + i * 3 + 2]; // Z-component

            nodes.emplace_back(node);
            // std::cerr << "Loaded " << i << "\n" << node << "\n";
        }

        quadratic_tets.emplace_back(nodes);

        // std::cerr << "Loaded " << quadratic_tets.size() << " quadratic tetrahedrons" << "\n";
    }
}



IntersectionOutput intersect_bvh_quad_tet(const Ray& ray,
    std::vector<Quadratic_tet> quadratic_tets,
    const unsigned int bvh_node_quad_tet_count) {

    // Go through all the tetrahedron and find an intersection of each triangle with a ray

    // Define default negative output if there is no intersection
    IntersectionOutput negative_output{
        Eigen::ArrayXXd(bvh_node_quad_tet_count, 3),
        EiVectorD3d::Zero(bvh_node_quad_tet_count, 3),
        Eigen::Vector<double, Eigen::Dynamic>::Constant(bvh_node_quad_tet_count, 1, std::numeric_limits<double>::infinity())
    };

    // Calculations - go through all the tetrahedrons
    
    // Convert between 2 types of rays, need to get rid of this later
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    Ray_old ray_old(a, b);
    ray_convert(ray_old, ray);


    EiVectorD3d plane_normals(bvh_node_quad_tet_count, 3);
    Eigen::ArrayXXd t_values(bvh_node_quad_tet_count, 1);
    Eigen::ArrayXXd barycentric_u(bvh_node_quad_tet_count, 1);
    Eigen::ArrayXXd barycentric_v(bvh_node_quad_tet_count, 1);

    for (int quad_tet_idx = 0; quad_tet_idx < bvh_node_quad_tet_count; quad_tet_idx++) {
        Eigen::Vector3d n_tmp;
        Eigen::Vector2d uv_tmp;
        double t = quadratic_tets[quad_tet_idx].intersect(ray_old, n_tmp, uv_tmp);

        // Convert the intersection results to acceptable format
        for (int i = 0; i < 3; ++i) {
            plane_normals(quad_tet_idx, i) = n_tmp(i);
        }
        
        t_values(quad_tet_idx) = t;
        barycentric_u(quad_tet_idx) = uv_tmp.x();
        barycentric_v(quad_tet_idx) = uv_tmp.y();
        
    }

    // Mask inappropriate values based on t_values
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> valid_mask;
    valid_mask = (t_values > 0.0); // t=0.0 means no intersection with the tet
    if (!valid_mask.any()) {
        //std::cout << "Condition 1 triggered" << std::endl;
        return negative_output; // No intersection - return infinity
    }

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
    Eigen::ArrayXXd barycentric_coordinates(bvh_node_quad_tet_count, 3);
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

            
            // std::cout << '\n' << Node.nodes_per_element << '\n';
            // std::cout << "The size of node_coords is: " << Node.node_coords.size() << '\n' << '\n';

            if (Node.nodes_per_element == TRI3) {
                out_intersection = intersect_bvh_triangles(ray, Node.node_coords, Node.element_count);
            }
            else if (Node.nodes_per_element == TET10) {
                std::vector<Quadratic_tet> quadratic_tets;
                load_quad_tets(Node.node_coords, quadratic_tets, Node.element_count, Node.nodes_per_element);
                // IntersectionOutput out_intersection_dum = intersect_bvh_quad_tet(ray, quadratic_tets, Node.element_count);
                // out_intersection = intersect_bvh_triangles(ray, Node.node_coords, Node.element_count);
                out_intersection = intersect_bvh_quad_tet(ray, quadratic_tets, Node.element_count);
            }


            // TEST
            // Ray ray_dum;
            // std::vector<double> node_coords_dum; 
            // unsigned int bvh_node_triangle_quad_count;
            // std::vector<Quadratic_tet> quadratic_tets; 
            // IntersectionOutput out_intersection_dum = intersect_bvh_quad_tet(ray_dum, quadratic_tets, bvh_node_triangle_quad_count);


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