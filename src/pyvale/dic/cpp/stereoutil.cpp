// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#define _USE_MATH_DEFINES
#include <cmath>

// dic header files
#include "./stereoutil.hpp"
#include "./dicfourier.hpp"
#include "./dicresults.hpp"
#include "./dicinterp.hpp"
#include "./dicsubset.hpp"

// calib header files
#include "../../calib/cpp/calibstereo.hpp"

// Eigen Header files
#include <Eigen/Core>

namespace stereo {



    Geometry compute_stereo_geometry(const Calib &calib) {

        common_util::Timer timer("to compute stereo geometry:", 2);

        Geometry geom;
        geom.K0  = camera_matrix(calib.cam0);
        geom.K1  = camera_matrix(calib.cam1);
        geom.t_x = skew_translation(calib.translation);
        geom.R   = rotation_from_euler(calib.rotation);
        geom.F   = fundamental(geom.K0, geom.K1, geom.t_x, geom.R);
        return geom;
    }

    void pixel_to_world(const subset::Grid &ss_grid,
                        const Calib &calib,
                        ResultArrays &temporal,
                        ResultArrays &stereo_ref,
                        ResultArrays &stereo_def,
                        const Eigen::Matrix3d &K0,
                        const Eigen::Matrix3d &K1,
                        const Eigen::Matrix3d &R, 
                        const int ss_size,
                        const bool first_frame){

        Eigen::Vector3d t(calib.translation[0],calib.translation[1],calib.translation[2]);
        Eigen::Matrix<double,3,4> P0, P1;

        P0 << Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero();
        P1 << R, t;   // just gonna assume t is in mm for now
    

        #pragma omp parallel for
        for (int ss = 0; ss < ss_grid.num; ss++){
            
            if ((!first_frame && !temporal.above_thresh[ss]) || !stereo_def.above_thresh[ss])
                continue;

            // centre coords left
            double cx_l = ss_grid.coords[ss*2] + temporal.u[ss];
            double cy_l = ss_grid.coords[ss*2+1] + temporal.v[ss];

            // centre coords right
            double cx_r = ss_grid.coords[ss*2] + stereo_def.u[ss];
            double cy_r = ss_grid.coords[ss*2+1] + stereo_def.v[ss];
        
            // undistorted pixel value
            double u_cx_l, u_cx_r, u_cy_l, u_cy_r;
            stereo::undistortPoint(u_cx_l, u_cy_l, cx_l, cy_l, K0, calib.cam0.distortion);
            stereo::undistortPoint(u_cx_r, u_cy_r, cx_r, cy_r, K1, calib.cam1.distortion);

            // 3d pixel coords guess
            Eigen::Vector3d xl(u_cx_l, u_cy_l, 1.0);
            Eigen::Vector3d xr(u_cx_r, u_cy_r, 1.0);

            // Build DLT system
            Eigen::Matrix4d A;

            A.row(0) = xl(0) * P0.row(2) - P0.row(0);
            A.row(1) = xl(1) * P0.row(2) - P0.row(1);
            A.row(2) = xr(0) * P1.row(2) - P1.row(0);
            A.row(3) = xr(1) * P1.row(2) - P1.row(1);

            // Solve with SVD
            Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
            Eigen::Vector4d X = svd.matrixV().col(3);

            // Convert from homogeneous
            X /= X(3);

            double X_mm = X(0);
            double Y_mm = X(1);
            double Z_mm = X(2);

            stereo_def.x_world[ss] = X_mm;
            stereo_def.y_world[ss] = Y_mm;
            stereo_def.z_world[ss] = Z_mm;

            if (first_frame) {
                stereo_def.u_world[ss] = 0.0;
                stereo_def.v_world[ss] = 0.0;
                stereo_def.w_world[ss] = 0.0;
            }
            else {
                // compute delta relative to the provided stereo_ref world coords
                // and add any previously-accumulated world displacement stored in
                // stereo_ref.*_world. This ensures cumulative displacements are
                // preserved when the reference is updated incrementally.
                stereo_def.u_world[ss] = (X_mm - stereo_ref.x_world[ss]) + stereo_ref.u_world[ss];
                stereo_def.v_world[ss] = (Y_mm - stereo_ref.y_world[ss]) + stereo_ref.v_world[ss];
                stereo_def.w_world[ss] = (Z_mm - stereo_ref.z_world[ss]) + stereo_ref.w_world[ss];
            }
        }
        if (first_frame) stereo_ref = stereo_def;
    }

    void undistortPoint(double &x_undistorted, double &y_undistorted,
                        const double x_distorted, const double y_distorted,
                        const Eigen::Matrix3d &K,
                        const std::vector<double> &d) {

        const double fx = K(0,0);
        const double fy = K(1,1);
        const double fs = K(0,1);
        const double cx = K(0,2);
        const double cy = K(1,2);
        const double k1 = d[0];
        const double k2 = d[1];
        const double p1 = d[2];
        const double p2 = d[3];
        const double k3 = d[4];

        // Normalize to camera coords
        double y_n = (y_distorted - cy) / fy;
        double x_n = (x_distorted - cx - fs * y_n) / fx;

        // Initial guess = normalized distorted point
        double x_u = x_n;
        double y_u = y_n;

        for (int i = 0; i < 100; i++) {
            double r2 = x_u*x_u + y_u*y_u;
            double r4 = r2*r2;
            double r6 = r2*r4;

            double radial = 1.0 + k1*r2 + k2*r4 + k3*r6;

            // Tangential distortion terms
            double dx = 2.0*p1*x_u*y_u       + p2*(r2 + 2.0*x_u*x_u);
            double dy = p1*(r2 + 2.0*y_u*y_u) + 2.0*p2*x_u*y_u;

            // Predicted distorted point
            double x_pred = x_u*radial + dx;
            double y_pred = y_u*radial + dy;

            double ex = x_n - x_pred;
            double ey = y_n - y_pred;

            if (std::abs(ex) < 1e-12 && std::abs(ey) < 1e-12)
                break;

            double drad_dr2 = k1 + 2.0*k2*r2 + 3.0*k3*r4;
            double drad_dxu = 2.0*x_u*drad_dr2;
            double drad_dyu = 2.0*y_u*drad_dr2;

            double J00 = radial + x_u*drad_dxu + 2.0*p1*y_u + 6.0*p2*x_u;
            double J01 =          x_u*drad_dyu + 2.0*p1*x_u + 2.0*p2*y_u;
            double J10 =          y_u*drad_dxu + 2.0*p1*x_u + 2.0*p2*y_u;
            double J11 = radial + y_u*drad_dyu + 6.0*p1*y_u + 2.0*p2*x_u;

            // Solve 2x2
            double det = J00*J11 - J01*J10;
            if (std::abs(det) < 1e-14)
                break;  // singular, shouldn't happen in practice

            x_u += (J11*ex - J01*ey) / det;
            y_u += (J00*ey - J10*ex) / det;
        }

        x_undistorted = x_u;
        y_undistorted = y_u;
    }

    void search_epi_line(double &best_zncc, 
                         double &best_disp_x, 
                         double &best_disp_y,
                         const double x,
                         const double y,
                         const subset::Pixels &ss_l,
                         subset::Pixels &ss_r,
                         const Eigen::Vector2d P,
                         const Eigen::Vector2d dir,
                         const Interpolator &interp_r,
                         const int range){


        

        best_zncc = -1.0;
        best_disp_x = 0.0;
        best_disp_y = 0.0;

        double corner_x, corner_y, zncc;
        Eigen::Vector2d P_i;

        for (int i = -range; i < range; i++){

            P_i = P + static_cast<double>(i)*dir;

            // Convert to CORNER position for get_subpx_from_img
            subset::get_corner(corner_x,corner_y, P_i(0),P_i(1),ss_l.size_x,ss_l.size_y);

            subset::fill_from_img_subpx(ss_r, corner_x, corner_y, interp_r);

            zncc = subset::zncc(ss_l, ss_r);

            if (zncc > best_zncc) {
                best_zncc = zncc;
                best_disp_x = corner_x - x;
                best_disp_y = corner_y - y;
            }
        }
    }



    void compute_epi(Eigen::Vector2d &nearest_point, 
                     Eigen::Vector2d &direction, 
                     const double x,
                     const double y,
                     const Eigen::Matrix3d &F){



            // std::cout << "subset" << std::endl;
            // std::cout << x << " " << y << std::endl;
            Eigen::Vector3d epi_line = F * Eigen::Vector3d(x, y, 1.0);

            // std::cout << "epi_line" << std::endl;
            // std::cout << epi_line << std::endl;

            double a = epi_line(0), b = epi_line(1), c = epi_line(2);
            double denom = a*a + b*b;
            double nrm = std::sqrt(denom);
            double t = (a*x + b*y + c)/denom;

            // std::cout << "t" << std::endl;
            // std::cout << t << std::endl;

            nearest_point << x - a*t, y - b*t;
            direction << -b/nrm, a/nrm;

            // std::cout << "nearest_point" << std::endl;
            // std::cout << nearest_point << std::endl;
            //
            if (direction(0) < 0)
                direction = -direction;
    }

    void get_rigid_translation_from_rectified_fft(std::vector<double> &p,
                                                  const double cx, const double cy,
                                                  const int ss_size_x, const int ss_size_y,
                                                  const int window_size_x, const int window_size_y,
                                                  const Eigen::Matrix3d &F,
                                                  const Interpolator &interp_ref,
                                                  const Interpolator &interp_def,
                                                  const double offset_x,
                                                  const double offset_y,
                                                  const bool print){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;
        const int window_half_x = window_size_x/2;
        const int window_half_y = window_size_y/2;
        const int ss_half_x = ss_size_x/2;
        const int ss_half_y = ss_size_y/2;

        // class for FFT
        FFT fft(window_size_x, window_size_y, print);

        // put the subset at the corner of the window.
        // for the FFT I'm just using a square subset and not the shape function
        // parameters. I've not found a case where this has been insufficient
        fill_fft_window_with_subset_at_centre(fft.ss_ref, interp_ref,
                                              cx, cy, px_hori, px_vert,
                                              ss_size_x, ss_size_y,
                                              window_size_x, window_size_y);

        // equation of epipolar line for the corner
        Eigen::Vector2d closest_point, dir;
        stereo::compute_epi(closest_point, dir, cx + offset_x, cy + offset_y, F);



        // TODO: Add a proper flag for this 
        bool subpx = true;

        Eigen::Vector2d perp(dir(1), -dir(0));

        for(int y = 0; y < window_size_y; y++){
            for(int x = 0; x < window_size_x; x++) {
                Eigen::Vector2d centre = closest_point + (x-window_half_x)*dir;
                Eigen::Vector2d sample_pt = centre - (y-window_half_y)*perp;
                double val = interp_def.eval(0,0,sample_pt(0),sample_pt(1));
                const int idx = y * window_size_x + x;
                if (fft.ss_def.has_coords()) {
                    fft.ss_def.x[idx] = sample_pt(0);
                    fft.ss_def.y[idx] = sample_pt(1);
                }
                fft.ss_def.vals[idx] = val;
            }
        }

        // apply window to deformed subset
        for (int row = 0; row < window_size_y; ++row) {
            for (int col = 0; col < window_size_x; ++col) {
                double coeff = 1.0; //fourier::hamming(row,col,window_size_x, window_size_y);
                fft.ss_def.vals[row*window_size_x+col] *= coeff;
            }
        }

        // zero norm the subsets
        bool normed_ref = fft.zero_norm_subsets_centered(fft.ss_ref, ss_size_x,ss_size_y, window_size_x, window_size_y);
        bool normed_def = fft.zero_norm_subset(fft.ss_def, window_size_x,window_size_y);

        // get peaks from the cross correlation
        double max_val = 0.0, peak_x = 0.0, peak_y = 0.0;
        if (normed_ref && normed_def){
            fft.correlate();
            fft.get_peak(peak_x, peak_y, max_val, subpx, "GAUSSIAN_2D");
        }
        //std::cout << "peak: " << peak_x << " " << peak_y << std::endl;

        // coordinate transform
        // peak_x = peak_x - window_half_x;
        // peak_y = peak_y - window_half_y;

        if (print) {
            for (int row = 0; row < window_size_y; ++row) {
                for (int col = 0; col < window_size_x; ++col) {
                    int idx  = row*window_size_x+col;
                    std::cout << col << " " << row << " ";
                    std::cout << fft.ss_ref.x[idx] << " " << fft.ss_ref.y[idx] << " " << fft.ss_ref.vals[idx] << " ";
                    std::cout << fft.ss_def.x[idx] << " " << fft.ss_def.y[idx] << " " << fft.ss_def.vals[idx] << " ";
                    std::cout << fft.cross_corr[idx] << std::endl;
                }
            }
            std::cout << std::endl;
        }


        // Compute the unrectified position in the right image
        Eigen::Vector2d unrectified_pos = closest_point + peak_x * dir - peak_y * perp;

        //std::cout << "unrectified_pos: " << unrectified_pos(0) << " " << unrectified_pos(1) << std::endl;
        p[0] = unrectified_pos(0) - cx;
        p[1] = unrectified_pos(1) - cy;
        p[2] = dir(0) - 1.0;
        p[3] = -perp(0);
        p[4] = dir(1);
        p[5] = -perp(1) - 1.0;

    }

   void get_rigid_translation_from_rectified_search(std::vector<double> &p,
                                                    const int ss_x, const int ss_y,
                                                    const int ss_size_x, const int ss_size_y,
                                                    const Eigen::Vector2d closest_point,
                                                    const Eigen::Vector2d dir,
                                                    subset::Pixels &ss_l,
                                                    const Interpolator &interp_ref,
                                                    const Interpolator &interp_def){

        const int px_hori = interp_def.px_hori;
        const int px_vert = interp_def.px_vert;

        int range = 100;
        Eigen::Vector2d perp(dir(1), -dir(0));
        subset::Pixels ss_r(ss_size_x, ss_size_y);
        subset::Pixels ss_final(ss_size_x, ss_size_y);
        double best_zncc = -1.0;

        //Optimizer opt_affine("AFFINE", "ZNSSD", 40, 0.0001, 0.90);

        for(int pt = -range; pt < range; pt++){


            Eigen::Vector2d def_coord = closest_point + (double(pt))*dir;

            // fill the deformed subset
            for(int y = 0; y < ss_size_y; y++){
                for(int x = 0; x < ss_size_x; x++) {
                    Eigen::Vector2d corner = def_coord + x*dir;
                    Eigen::Vector2d sample_pt = corner - y*perp;
                    double val = interp_def.eval(0,0,sample_pt(0),sample_pt(1));
                    
                    int idx = y*ss_size_x+x;
                    ss_r.x[idx] = sample_pt(0);
                    ss_r.y[idx] = sample_pt(1);
                    ss_r.vals[idx] = val;
                    // std::cout << ss_l.x[idx] << " " << ss_l.y[idx] << " " << ss_l.vals[idx] << " ";
                    // std::cout << ss_r.x[idx] << " " << ss_r.y[idx] << " " << ss_r.vals[idx] << std::endl;
                }
            }
            double zncc = subset::zncc(ss_l,ss_r);

            
            // testing with optimizer here
            // subset::get_subpx_from_img(ss_l, ss_x, ss_y, interp_ref);
            // opt_affine.p[0] = def_coord(0) - ss_x;
            // opt_affine.p[1] = def_coord(1) - ss_y;
            // opt_affine.p[2] = dir(0) - 1.0;
            // opt_affine.p[3] = -perp(0);
            // opt_affine.p[4] = dir(1);
            // opt_affine.p[5] = -perp(1) - 1.0;
            // OptResult seed_res = opt_affine.solve(ss_x, ss_y, ss_l, ss_r, interp_def);
            // std::cout << seed_res.cost << " " << int(seed_res.converged) << " " << seed_res.iter << std::endl;

            if (zncc > best_zncc) {
                best_zncc = zncc;
                p[0] = def_coord(0) - ss_x;
                p[1] = def_coord(1) - ss_y;
                p[2] = dir(0) - 1.0;
                p[3] = -perp(0);
                p[4] = dir(1);
                p[5] = -perp(1) - 1.0;
                // for (int px = 0; px < ss_r.num_px; px++){
                //     ss_final.x[px] = ss_r.x[px];
                //     ss_final.y[px] = ss_r.y[px];
                //     ss_final.vals[px] = ss_r.vals[px];
                // }
            }

        }

        // for (int px = 0; px < ss_r.num_px; px++){
        //     std::cout << ss_final.x[px] << " " << ss_final.y[px] << " " << ss_final.vals[px] << std::endl;
        // }
    }




    Eigen::Matrix3d rotation_from_euler(const std::vector<double> &rot_deg) {
        double theta = rot_deg[0] * (M_PI / 180.0);
        double phi   = rot_deg[1] * (M_PI / 180.0);
        double psi   = rot_deg[2] * (M_PI / 180.0);

        Eigen::Matrix3d Rx, Ry, Rz;
        Rx << 1, 0, 0,
            0, cos(theta), -sin(theta),
            0, sin(theta), cos(theta);

        Ry << cos(phi), 0, sin(phi),
            0, 1, 0,
            -sin(phi), 0, cos(phi);

        Rz << cos(psi), -sin(psi), 0,
            sin(psi), cos(psi), 0,
            0, 0, 1;

        return Rz * Ry * Rx;
    }

    Eigen::Matrix3d camera_matrix(const CamIntrinsics &cam) {
        Eigen::Matrix3d K;
        K << cam.fx, cam.fs, cam.cx,
            0.0, cam.fy, cam.cy,
            0.0, 0.0, 1.0;
        return K;
    }

    Eigen::Matrix3d skew_translation(const std::vector<double> &t) {
        Eigen::Matrix3d t_x;
        t_x << 0, -t[2], t[1],
            t[2], 0, -t[0],
            -t[1], t[0], 0;
        return t_x;
    }

    Eigen::MatrixXd patch_corners(const double x, 
                                  const double y, 
                                  const double size_x, 
                                  const double size_y) {
        Eigen::MatrixXd pts(3,4);
        pts << x,     x,       x+size_x, x+size_x,
            y,     y+size_y, y+size_y, y,
            1.0,   1.0,     1.0,          1.0;
        return pts;
    }

    Eigen::Matrix3d fundamental(const Eigen::Matrix3d &K0,
                                const Eigen::Matrix3d &K1,
                                const Eigen::Matrix3d &t_x,
                                const Eigen::Matrix3d &R){

        Eigen::Matrix3d K0_inv = K0.inverse();
        Eigen::Matrix3d K1_inv_T = K1.inverse().transpose();
        Eigen::Matrix3d F = K1_inv_T * t_x * R * K0_inv;
        F /= F.norm();
        return F;

    }

    EpipolarStrip calc_epi_strip_points(const double cx,
                                        const double cy,
                                        const Eigen::Matrix<double,3,4> &lines,
                                        const Eigen::Matrix3d F,
                                        const double range) {


        // equation of epipolar line 
        Eigen::Vector3d F_centre = F * Eigen::Vector3d(cx, cy, 1.0);
        double a = F_centre(0), b = F_centre(1), c = F_centre(2);

        double denom = a*a + b*b;
        double nrm = std::sqrt(denom);

        // bounds
        double cmin = std::numeric_limits<double>::infinity();
        double cmax = -std::numeric_limits<double>::infinity();
        
        // get the range of the bounding box
        for (int i = 0; i < 4; i++){
            double ai = lines(0,i);
            double bi = lines(1,i);
            double ci = lines(2,i);
            double offset = (ai*cx + bi*cy + ci)/nrm;
            cmin = std::min(cmin, offset);
            cmax = std::max(cmax, offset);
        }

        double half_width = 0.5*(cmax-cmin);

        double t = (a*cx + b*cy + c)/denom;
        Eigen::Vector2d P(cx - a*t, cy - b*t);
        Eigen::Vector2d dir(-b/nrm, a/nrm);
        Eigen::Vector2d n(a/nrm, b/nrm);


        // point on the epipolar line <range> pixels either side
        // of the midpoint P.
        Eigen::Vector2d P0 = P - range*dir;
        Eigen::Vector2d P1 = P + range*dir;

        return EpipolarStrip {
            P,                  // patch centre
            P0 + half_width*n,  // bounding box Q0
            P0 - half_width*n,  // bounding box Q1
            P1 - half_width*n,  // bounding box Q2
            P1 + half_width*n   // bounding box Q3
        };
    }


    std::tuple<int,int,int,int> bounding_box(const EpipolarStrip &strip, 
                                             const int px_hori, 
                                             const int px_vert){

        double xmin = std::max(0.0, std::min({strip.Q0.x(), strip.Q1.x(), strip.Q2.x(), strip.Q3.x()}));
        double xmax = std::min((double)px_hori-1.0, std::max({strip.Q0.x(), strip.Q1.x(), strip.Q2.x(), strip.Q3.x()}));
        double ymin = std::max(0.0, std::min({strip.Q0.y(), strip.Q1.y(), strip.Q2.y(), strip.Q3.y()}));
        double ymax = std::min((double)px_vert-1.0, std::max({strip.Q0.y(), strip.Q1.y(), strip.Q2.y(), strip.Q3.y()}));

        return {std::ceil(xmin), std::ceil(xmax), std::ceil(ymin), std::ceil(ymax)};
    }



    bool* compute_roi_r(const subset::Grid ss_grid_l,
                        const ResultArrays &stereo_matches,
                        const int px_hori,
                        const int px_vert,
                        const int ss_size_x,
                        const int ss_size_y) {

       
        const int half_y = ss_size_y/2;
        const int half_x = ss_size_x/2;

        bool* img_roi_r = new bool[px_hori * px_vert]();

        int ss_x_l, ss_y_l;
        double ss_x_r, ss_y_r;
        double cx, cy;
        for (int i = 0; i < stereo_matches.u.size(); i++) {
            
            ss_x_l = ss_grid_l.coords[2*i];
            ss_y_l = ss_grid_l.coords[2*i+1];
            subset::get_centre(cx, cy, ss_x_l, ss_y_l, ss_size_x, ss_size_y);

            ss_x_r = ss_x_l+stereo_matches.u[i];
            ss_y_r = ss_y_l+stereo_matches.v[i];

            for (int dy = -half_y; dy <= half_y; dy++) {
                for (int dx = -half_x; dx <= half_x; dx++) {
                    int x = static_cast<int>(std::round(ss_x_r)) + dx;
                    int y = static_cast<int>(std::round(ss_y_r)) + dy;
                    if (x >= 0 && x < px_hori && y >= 0 && y < px_vert) {
                        img_roi_r[y * px_hori + x] = true;
                    }
                }
            }
        }

        return img_roi_r;
    }

   bool* compute_roi_r_test(const bool* img_roi_l,
                    const subset::Grid& ss_grid,
                    const ResultArrays& stereo_matches,
                    const int px_hori,
                    const int px_vert) {

    // Count valid matches first
    std::vector<int> valid;
    for (int i = 0; i < stereo_matches.u.size(); i++)
        if (stereo_matches.conv[i] && stereo_matches.above_thresh[i])
            valid.push_back(i);

    assert(valid.size() >= 4 && "Not enough valid matches to compute H");

    Eigen::MatrixXd A(valid.size() * 2, 9);

    for (int i = 0; i < valid.size(); i++) {
        const int idx = valid[i];
        
        const double cx = ss_grid.coords[2*idx];
        const double cy = ss_grid.coords[2*idx+1];

        const double xr = cx + stereo_matches.u[idx];
        const double yr = cy + stereo_matches.v[idx];

        A.row(2*i)   << cx, cy, 1,  0,  0, 0, -xr*cx, -xr*cy, -xr;
        A.row(2*i+1) <<  0,  0, 0, cx, cy, 1, -yr*cx, -yr*cy, -yr;
    }

    // Solve via SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    Eigen::Matrix3d H;
    H << h(0), h(1), h(2),
        h(3), h(4), h(5),
        h(6), h(7), h(8);

    // Warp left mask to right using H
    bool* img_roi_r = new bool[px_hori * px_vert]();

    for (int v = 0; v < px_vert; v++) {
        for (int u = 0; u < px_hori; u++) {
            if (!img_roi_l[v * px_hori + u]) continue;

            Eigen::Vector3d p = H * Eigen::Vector3d(u, v, 1.0);
            int u_r = static_cast<int>(std::round(p.x() / p.z()));
            int v_r = static_cast<int>(std::round(p.y() / p.z()));

            for (int dv = -1; dv <= 1; dv++) {
                for (int du = -1; du <= 1; du++) {
                    int u_n = u_r + du, v_n = v_r + dv;
                    if (u_n >= 0 && u_n < px_hori && v_n >= 0 && v_n < px_vert)
                        img_roi_r[v_n * px_hori + u_n] = true;
                }
            }
        }
    }
    return img_roi_r;
}

    void remove_unmatched_subsets(subset::Grid& ss_grid_l, 
                                  subset::Grid& ss_grid_r,
                                  const ResultArrays stereo_matches) {

        // Build list of subsets to keep
        std::vector<int> keep;
        for (int ss = 0; ss < ss_grid_l.num; ss++) {
            if (stereo_matches.above_thresh[ss])
                keep.push_back(ss);
        }

        // Build remap
        std::vector<int> remap(ss_grid_l.num, -1);
        for (int i = 0; i < keep.size(); i++)
            remap[keep[i]] = i;

        // Compact coords
        std::vector<double> new_coords_l, new_coords_r;

        for (int i = 0; i < keep.size(); i++) {
            int ss = keep[i];
            new_coords_l.push_back(ss_grid_l.coords[2*ss]);
            new_coords_l.push_back(ss_grid_l.coords[2*ss + 1]);
            new_coords_r.push_back(ss_grid_r.coords[2*ss]);
            new_coords_r.push_back(ss_grid_r.coords[2*ss + 1]);
        }

        // Compact neigh
        std::vector<std::vector<int>> new_neigh_l, new_neigh_r;
        for (int i = 0; i < keep.size(); i++) {
            int ss = keep[i];

            std::vector<int> nl, nr;
            for (int j = 0; j < ss_grid_l.neigh[ss].size(); j++) {
                int n = ss_grid_l.neigh[ss][j];
                if (remap[n] != -1)
                    nl.push_back(remap[n]);
            }
            for (int j = 0; j < ss_grid_r.neigh[ss].size(); j++) {
                int n = ss_grid_r.neigh[ss][j];
                if (remap[n] != -1)
                    nr.push_back(remap[n]);
            }

            new_neigh_l.push_back(nl);
            new_neigh_r.push_back(nr);
        }

        // Fix mask
        for (int i = 0; i < (int)ss_grid_l.mask.size(); i++) {
            if (ss_grid_l.mask[i] != -1)
                ss_grid_l.mask[i] = remap[ss_grid_l.mask[i]];
            if (ss_grid_r.mask[i] != -1)
                ss_grid_r.mask[i] = remap[ss_grid_r.mask[i]];
        }

        ss_grid_l.coords = new_coords_l;
        ss_grid_l.neigh  = new_neigh_l;
        ss_grid_l.num    = keep.size();

        ss_grid_r.coords = new_coords_r;
        ss_grid_r.neigh  = new_neigh_r;
        ss_grid_r.num    = keep.size();
    }

    std::pair<std::vector<std::string>, std::vector<std::string>>
    split_basenames(const util::Config &conf) {
        if (!conf.stereo)
            return {conf.basenames, {}};

        std::vector<std::string> basenames_l, basenames_r;
        for (int i = 0; i < conf.basenames.size() / 2; i++) {
            basenames_l.push_back(conf.basenames[i]);
            basenames_r.push_back(conf.basenames[conf.num_def_img + 1 + i]);
        }
        return {basenames_l, basenames_r};
    }


}


