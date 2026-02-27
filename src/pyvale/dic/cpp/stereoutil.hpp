// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef _STEREOUTIL_HPP
#define _STEREOUTIL_HPP


// dic header files
#include "./dicsubset.hpp"
#include "./dicresults.hpp"

// calibration header files
#include "../../calib/cpp/calibstereo.hpp"

// Eigen Header files
#include <Eigen/Core>


namespace stereo {



    /**
    * @brief Represents the epipolar strip bounding geometry for a patch.
    *
    * Stores the projected patch centre on the epipolar line and the four
    * corner points of the bounding quadrilateral that limits the valid
    * search region around that line.
    */
    struct EpipolarStrip {
        Eigen::Vector2d P;   ///< Patch centre projected onto the epipolar line.
        Eigen::Vector2d Q0;  ///< Bounding corner Q0.
        Eigen::Vector2d Q1;  ///< Bounding corner Q1.
        Eigen::Vector2d Q2;  ///< Bounding corner Q2.
        Eigen::Vector2d Q3;  ///< Bounding corner Q3.
    };

    /**
    * @brief Container for stereo camera geometry derived from calibration.
    *
    * Holds intrinsic matrices for both cameras, the rotation and translation
    * between them (in skew-symmetric form), and the resulting fundamental matrix.
    */
    struct Geometry {
        Eigen::Matrix3d K0;   ///< Intrinsic matrix of camera 0.
        Eigen::Matrix3d K1;   ///< Intrinsic matrix of camera 1.
        Eigen::Matrix3d R;    ///< Rotation from camera 0 to camera 1.
        Eigen::Matrix3d t_x;  ///< Skew-symmetric translation matrix.
        Eigen::Matrix3d F;    ///< Fundamental matrix.
    };



    /**
    * @brief Computes all stereo geometry matrices from calibration parameters.
    *
    * Builds the intrinsic matrices of both cameras, the rotation and translation
    * between them (in skew-symmetric form), and the resulting fundamental matrix.
    *
    * @param calib Stereo calibration parameters (intrinsics, rotation, translation).
    * @return Geometry struct containing K0, K1, R, t_x, and F.
    */
    Geometry compute_stereo_geometry(const Calib &calib);

    void undistortPoint(double &x_undistorted, double  &y_undistorted,
                        const double x_distorted, const double y_distorted,
                        const Eigen::Matrix3d &K,
                        const std::vector<double> &d);


    /**
    * @brief Computes a rotation matrix from Euler angles (degrees, XYZ order).
    * @param rot_deg Euler angles in degrees: [theta_x, phi_y, psi_z].
    * @return 3x3 rotation matrix Rz * Ry * Rx.
    */
    Eigen::Matrix3d rotation_from_euler(const std::vector<double> &rot_deg);


    /**
    * @brief Constructs a camera intrinsic matrix from calibration parameters.
    * @param cam Camera intrinsics (fx, fy, fs, cx, cy).
    * @return 3x3 intrinsic matrix K.
    */
    Eigen::Matrix3d camera_matrix(const CamIntrinsics &cam);


    /**
    * @brief Forms the skew-symmetric matrix of a translation vector.
    * @param t Translation vector [tx, ty, tz].
    * @return 3x3 skew-symmetric matrix t_x.
    */
    Eigen::Matrix3d skew_translation(const std::vector<double> &t);




    /**
    * @brief Computes the homogeneous corner coordinates of an image patch.
    * @param x Top-left x coordinate.
    * @param y Top-left y coordinate.
    * @param size_x Patch width.
    * @param size_y Patch height.
    * @return 3x4 matrix of patch corner points.
    */
    Eigen::MatrixXd patch_corners(const double x, 
                                  const double y, 
                                  const double size_x, 
                                  const double size_y);
    /**
    * @brief Computes the fundamental matrix from intrinsics, translation, and rotation.
    * @param K0 Intrinsic matrix of camera 0.
    * @param K1 Intrinsic matrix of camera 1.
    * @param t_x Skew-symmetric translation matrix.
    * @param R Rotation from camera 0 to camera 1.
    * @return Normalized 3x3 fundamental matrix F.
    */
    Eigen::Matrix3d fundamental(const Eigen::Matrix3d &K0,
                                const Eigen::Matrix3d &K1,
                                const Eigen::Matrix3d &t_x,
                                const Eigen::Matrix3d &R);



    /**
    * @brief Calculates epipolar strip geometry around a patch centre.
    * @param cx Patch centre x coordinate.
    * @param cy Patch centre y coordinate.
    * @param lines Bounding box lines in homogeneous form.
    * @param F Fundamental matrix.
    * @param range Half-length of the epipolar search segment.
    * @return EpipolarStrip containing midpoint and bounding quad points.
    */
    EpipolarStrip calc_epi_strip_points(const double cx, const double cy,
                                               const Eigen::Matrix<double,3,4> &lines,
                                               const Eigen::Matrix3d F,
                                               double range);



    /**
    * @brief Computes the image-space bounding box of an epipolar strip.
    * @param strip Epipolar strip quad points.
    * @param px_hori Image width in pixels.
    * @param px_vert Image height in pixels.
    * @return (xmin, xmax, ymin, ymax) bounding coordinates.
    */
    std::tuple<int,int,int,int> bounding_box(const EpipolarStrip &strip, 
                                             const int px_hori, 
                                             const int px_vert);





    /**
    * @brief Triangulates 3D world coordinates for valid stereo matches (linear DLT).
    *
    * For each subset center whose match passes the threshold, this function:
    *  1) Computes the left/right patch center in pixels,
    *  2) Undistorts both points using each camera's intrinsics/distortion,
    *  3) Forms normalized homogeneous points (assumes K = I in projection),
    *  4) Builds the DLT system with P0 = [I | 0], P1 = [R | t],
    *  5) Solves via SVD and recovers X in homogeneous coordinates.
    *
    * @param ss_grid Grid of subset (patch) top-left coordinates (pixel space).
    * @param calib   Calibration containing intrinsics/distortion for cam0/cam1 and translation (t).
    * @param stereo_matches Match results; uses u,v (pixel disparities) and above_thresh mask.
    * @param K0      Intrinsic matrix of camera 0 (used only for undistortion here).
    * @param K1      Intrinsic matrix of camera 1 (used only for undistortion here).
    * @param R       Rotation from camera 0 to camera 1.
    * @param ss_size Subset (patch) size in pixels (assumed square).
    *
    * @note Points are undistorted and treated as normalized image coordinates when building
    *       the DLT system (i.e., P0 = [I|0], P1 = [R|t]). Ensure undistortion produces
    *       normalized coordinates consistent with this assumption.
    * @note Translation vector @p calib.translation is assumed to be in millimeters, so the
    *       resulting (X, Y, Z) will also be in millimeters.
    * @pre  ss_grid.num == stereo_matches.u.size() == stereo_matches.v.size()
    *       == stereo_matches.above_thresh.size()
    * @warning This implementation computes X but does not store it; add assignments to
    *          persist (X_mm, Y_mm, Z_mm) into your results container as needed.
    *          TODO: NEED TO FIX THIS
    */
    void pixel_to_world(const subset::Grid &ss_grid,
                        const Calib &calib,
                        ResultArrays &stereo_matches,
                        const Eigen::Matrix3d K0,
                        const Eigen::Matrix3d K1,
                        const Eigen::Matrix3d R, 
                        const int ss_size);



    /**
    * @brief Searches along an epipolar line for the best ZNCC match.
    * @param best_zncc Output: highest ZNCC score found.
    * @param best_disp_x Output: best displacement in x.
    * @param best_disp_y Output: best displacement in y.
    * @param x Reference patch centre x.
    * @param y Reference patch centre y.
    * @param ss_l Reference subset (left image).
    * @param ss_r Workspace subset (right image).
    * @param P Midpoint on the epipolar line.
    * @param dir Direction of the epipolar line (unit length).
    * @param interp_r Interpolator for the right image.
    * @param range Search extent in pixels along the line.
    */
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
                         const int range);


    /**
    * @brief Computes the closest point and direction of the epipolar line for a pixel.
    * @param nearest_point Output: orthogonal projection of (x,y) onto the epipolar line.
    * @param direction Output: unit direction vector of the epipolar line.
    * @param x Pixel x coordinate.
    * @param y Pixel y coordinate.
    * @param F Fundamental matrix.
    */
    void compute_epi(Eigen::Vector2d &nearest_point,
                     Eigen::Vector2d &direction,
                     const double x,
                     const double y,
                     const Eigen::Matrix3d F);

 

    /**
    * @brief Estimates rigid translation using FFT correlation on a rectified search grid.
    * @param p Output: 6‑parameter rigid/affine displacement seed.
    * @param ss_x Subset top‑left x coordinate.
    * @param ss_y Subset top‑left y coordinate.
    * @param ss_size_x Subset width.
    * @param ss_size_y Subset height.
    * @param closest_point Epipolar closest point to the reference pixel.
    * @param dir Epipolar direction unit vector.
    * @param window_size_x FFT window width.
    * @param window_size_y FFT window height.
    * @param img_ref Pointer to reference image buffer.
    * @param interp_def Interpolator for the deformed image.
    */
    void get_rigid_translation_from_rectified_fft(std::vector<double> &p,
                                                  const int ss_x, const int ss_y,
                                                  const int ss_size_x, const int ss_size_y,
                                                  const Eigen::Vector2d closest_point,
                                                  const Eigen::Vector2d dir,
                                                  const int window_size_x, const int window_size_y,
                                                  const double *img_ref,
                                                  const Interpolator &interp_def);


    /**
    * @brief Estimates rigid translation by brute‑force ZNCC search along the epipolar line.
    * @param p Output: 6‑parameter rigid/affine displacement seed.
    * @param ss_x Subset top‑left x coordinate.
    * @param ss_y Subset top‑left y coordinate.
    * @param ss_size_x Subset width.
    * @param ss_size_y Subset height.
    * @param closest_point Epipolar closest point to the reference pixel.
    * @param dir Epipolar direction unit vector.
    * @param ss_l Reference subset (left image).
    * @param interp_ref Interpolator for reference image.
    * @param interp_def Interpolator for deformed image.
    */
   void get_rigid_translation_from_rectified_search(std::vector<double> &p,
                                                    const int ss_x, const int ss_y,
                                                    const int ss_size_x, const int ss_size_y,
                                                    const Eigen::Vector2d closest_point,
                                                    const Eigen::Vector2d dir,
                                                    subset::Pixels &ss_l,
                                                    const Interpolator &interp_ref,
                                                    const Interpolator &interp_def);
}



#endif // _STEREOUTIL_HPP
