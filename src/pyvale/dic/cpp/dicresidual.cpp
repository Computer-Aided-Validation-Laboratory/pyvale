// ================================================================================
// pyvale: the python validation engine
// License: MIT
// ================================================================================

#include "./dicresidual.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include "../../commoncpp/csvbuffer.hpp"
#include "./dicinterpfactory.hpp"

double sample(const Interpolator &image, double x, double y) {
    if (x < 0.0 || y < 0.0 ||
        x > static_cast<double>(image.px_hori - 1) ||
        y > static_cast<double>(image.px_vert - 1)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return image.eval(0, 0, x, y);
}

void calculate_illumination_fields(std::vector<double> &a, // brightness
                                   std::vector<double> &b, // contrast
                                   std::vector<double> &ref_std,
                                   std::vector<uint8_t> &valid,
                                   const Interpolator &reference,
                                   const Interpolator &deformed,
                                   const SubsetGrid &subset_grid,
                                   const ResultArrays &results,
                                   int subset_size,
                                   util::EShapeFunc shape_func) {

    // "ABSOLUTE":   R(x,y) = R(x,y) = f(x,y)−g(x′,y′)
    // "NORMALIZED": R(x,y) = a(x,y)+b(x,y)×f(x,y)−g(x′,y′).

    const int num_points = subset_grid.num;

    a.assign(num_points, std::numeric_limits<double>::quiet_NaN());
    b.assign(num_points, std::numeric_limits<double>::quiet_NaN());
    ref_std.assign(num_points, std::numeric_limits<double>::quiet_NaN());
    valid.assign(num_points, 0);

    if (subset_size <= 0 || subset_size % 2 == 0) return;

    #pragma omp parallel for schedule(static) if (num_points > 32)
    for (int i = 0; i < num_points; ++i) {
        if (!results.valid_match[i] ||
            !std::isfinite(results.u[i]) || !std::isfinite(results.v[i])) {
            continue;
        }

        const double cx = subset_grid.coords[2*i];
        const double cy = subset_grid.coords[2*i + 1];

        Subset<double, true> reference_subset(subset_size, subset_size);
        Subset<double, true> deformed_subset(subset_size, subset_size);
        reference_subset.fill_from_centre_coords(cx, cy, reference);

        bool use_shape_parameters =
            results.num_params > 0 &&
            results.p.size() >= static_cast<std::size_t>((i + 1) * results.num_params);

        std::vector<double> params;
        if (use_shape_parameters) {
            params.assign(results.p.begin() + i * results.num_params,
                          results.p.begin() + (i + 1) * results.num_params);
            use_shape_parameters = std::all_of(
                params.begin(), params.end(), [](double value) {
                    return std::isfinite(value);
                });
        }

        if (use_shape_parameters) {
            deformed_subset.fill_from_shape_params(
                cx, cy, params, deformed, shape_func);
        }
        else {
            params = {results.u[i], results.v[i]};
            deformed_subset.fill_from_shape_params(
                cx, cy, params, deformed, util::EShapeFunc::RIGID);
        }

        double mean_ref = 0.0;
        double mean_def = 0.0;
        int count = 0;
        for (int pixel = 0; pixel < reference_subset.num_px; ++pixel) {
            const double f = reference_subset.vals[pixel];
            const double g = deformed_subset.vals[pixel];
            if (std::isfinite(f) && std::isfinite(g)) {
                mean_ref += f;
                mean_def += g;
                ++count;
            }
        }
        if (count < 2) continue;

        mean_ref /= count;
        mean_def /= count;
        double ref_var = 0.0;
        double covar = 0.0;

        for (int pixel = 0; pixel < reference_subset.num_px; ++pixel) {

            const double f = reference_subset.vals[pixel];
            const double g = deformed_subset.vals[pixel];
            if (!std::isfinite(f) || !std::isfinite(g)) continue;
            const double df = f - mean_ref;
            const double dg = g - mean_def;
            ref_var += df * df;
            covar += df * dg;
        }

        const double var_floor = 1.0e-12;
        const double slope = ref_var > var_floor ? covar / ref_var : 1.0;

        a[i] = mean_def - slope * mean_ref;
        b[i] = slope;
        ref_std[i] = std::sqrt(ref_var / count);
        valid[i] = 1;

    }
}

void raster_triangle(std::vector<double> &R,
                     const Interpolator &interp_ref,
                     const Interpolator &interp_def,
                     const bool *roi_mask,
                     int a,
                     int b,
                     int c,
                     const std::vector<double> &coords,
                     const std::vector<double> &u,
                     const std::vector<double> &v,
                     const std::vector<uint8_t> &valid,
                     util::EResidualMode mode,
                     const std::vector<double> &brightness,
                     const std::vector<double> &contrast,
                     const std::vector<double> &reference_std,
                     const std::vector<uint8_t> &illumination_valid) {

    if (a < 0 || b < 0 || c < 0 ||
        !valid[a] || !valid[b] || !valid[c] ||
        (mode == util::EResidualMode::NORMALIZED &&
         (!illumination_valid[a] || !illumination_valid[b] || !illumination_valid[c]))) {
        return;
    }

    const double ax = coords[2*a];
    const double ay = coords[2*a + 1];
    const double bx = coords[2*b];
    const double by = coords[2*b + 1];
    const double cx = coords[2*c];
    const double cy = coords[2*c + 1];
    const double area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (std::abs(area) < 1.0e-12) {
        return;
    }

    const int x0 = std::max(0, static_cast<int>(std::ceil(std::min({ax, bx, cx}))));
    const int x1 = std::min(interp_ref.px_hori - 1, static_cast<int>(std::floor(std::max({ax, bx, cx}))));
    const int y0 = std::max(0, static_cast<int>(std::ceil(std::min({ay, by, cy}))));
    const int y1 = std::min(interp_ref.px_vert - 1, static_cast<int>(std::floor(std::max({ay, by, cy}))));

    #pragma omp parallel for schedule(static) if (y1 - y0 >= 8)
    for (int y_px = y0; y_px <= y1; ++y_px) {
        for (int x_px = x0; x_px <= x1; ++x_px) {

            const std::size_t pixel = static_cast<std::size_t>(y_px) * interp_ref.px_hori + x_px;

            if (!roi_mask[pixel] || !std::isnan(R[pixel])) {
                continue;
            }

            const double w_a = ((by - cy) * (x_px - cx) + (cx - bx) * (y_px - cy)) / area;
            const double w_b = ((cy - ay) * (x_px - cx) + (ax - cx) * (y_px - cy)) / area;
            const double w_c = 1.0 - w_a - w_b;

            constexpr double tol = 1.0e-10;
            if (w_a < -tol || w_b < -tol || w_c < -tol) {
                continue;
            }

            const double x_def = x_px + w_a * u[a] + w_b * u[b] + w_c * u[c];
            const double y_def = y_px + w_a * v[a] + w_b * v[b] + w_c * v[c];
            const double def_val = sample(interp_def, x_def, y_def);
            if (std::isfinite(def_val)) {
                const double ref_val = sample(interp_ref, x_px, y_px);
                if (mode == util::EResidualMode::NONE) {
                    R[pixel] = def_val - ref_val;
                }
                else if (mode == util::EResidualMode::ABSOLUTE) {
                    R[pixel] = std::abs(ref_val - def_val);
                }
                else {
                    const double brightness_val = w_a * brightness[a] + w_b * brightness[b] + w_c * brightness[c];
                    const double contrast_val = w_a * contrast[a] + w_b * contrast[b] + w_c * contrast[c];
                    const double sigma = w_a * reference_std[a] + w_b * reference_std[b] + w_c * reference_std[c];

                    if (std::isfinite(brightness_val) && std::isfinite(contrast_val) &&
                        std::isfinite(sigma) && sigma > 1.0e-12) {
                        R[pixel] = std::abs((brightness_val + contrast_val * ref_val) - def_val) / sigma;
                    }
                }
            }
        }
    }
}

void create_residual_map(std::vector<double> &R,
                         const Interpolator &interp_ref,
                         const Interpolator &interp_def,
                         const SubsetGrid &subset_grid,
                         const ResultArrays &results,
                         const bool *roi_mask,
                         int subset_size,
                         util::EShapeFunc shape_func,
                         util::EResidualMode mode) {


    // "ABSOLUTE":   R(x,y) = R(x,y) = f(x,y)−g(x′,y′)
    // "NORMALIZED": R(x,y) = a(x,y)+b(x,y)×f(x,y)−g(x′,y′).

    const int px_hori = interp_ref.px_hori;
    const int px_vert = interp_ref.px_vert;
    const int num_px = px_hori * px_vert;

    R.assign(num_px, std::numeric_limits<double>::quiet_NaN());

    if (subset_grid.num != results.num_ss ||
        subset_grid.coords.size() != static_cast<std::size_t>(2 * subset_grid.num)) {
        throw std::invalid_argument("Subset grid and residual result dimensions do not match");
    }


    std::vector<double> a; // brighness
    std::vector<double> b; // contrast
    std::vector<double> ref_std;
    std::vector<uint8_t> illumination_valid;
    if (mode == util::EResidualMode::NORMALIZED) {
        calculate_illumination_fields(a, b, ref_std,
                                      illumination_valid, interp_ref, interp_def,
                                      subset_grid, results, subset_size,
                                      shape_func);
    }

    // Triangulatation.
    for (int row = 0; row + 1 < subset_grid.num_ss_y; ++row) {
        for (int col = 0; col + 1 < subset_grid.num_ss_x; ++col) {

            const int tl = subset_grid.mask[row * subset_grid.num_ss_x + col];
            const int tr = subset_grid.mask[row * subset_grid.num_ss_x + col + 1];
            const int bl = subset_grid.mask[(row + 1) * subset_grid.num_ss_x + col];
            const int br = subset_grid.mask[(row + 1) * subset_grid.num_ss_x + col + 1];

            if (tl < 0 || tr < 0 ||
                bl < 0 || br < 0 ||
                !results.valid_match[tl] || !results.valid_match[tr] ||
                !results.valid_match[bl] || !results.valid_match[br] ||
                !std::isfinite(results.u[tl]) || !std::isfinite(results.v[tl]) ||
                !std::isfinite(results.u[tr]) || !std::isfinite(results.v[tr]) ||
                !std::isfinite(results.u[bl]) || !std::isfinite(results.v[bl]) ||
                !std::isfinite(results.u[br]) || !std::isfinite(results.v[br])) {
                continue;
            }

            raster_triangle(R, interp_ref, interp_def, roi_mask,
                            tl, tr, br,
                            subset_grid.coords, results.u, results.v,
                            results.valid_match, mode, a, b,
                            ref_std, illumination_valid);
            raster_triangle(R, interp_ref, interp_def, roi_mask,
                            tl, br, bl,
                            subset_grid.coords, results.u, results.v,
                            results.valid_match, mode, a, b,
                            ref_std, illumination_valid);
        }
    }
}

void write_residual_maps(const Interpolator &interp_ref_l,
                         const Interpolator &interp_def_l,
                         const Interpolator *interp_ref_r,
                         const Interpolator *interp_def_r,
                         const SubsetGrid &subset_grid,
                         const ResultArrays &results_def_l,
                         const ResultArrays *results_def_r,
                         const ResultArrays *results_ref_r,
                         const bool *roi_mask,
                         const util::Config &conf,
                         const common_util::SaveConfig &saveconf,
                         const std::string &left_image_filename,
                         const std::string &right_image_filename) {

    if (!saveconf.output_residual ||
        conf.residual_mode == util::EResidualMode::NONE) {
        return;
    }

    std::vector<double> residual_map;
    create_residual_map(residual_map,
                        interp_ref_l,
                        interp_def_l,
                        subset_grid,
                        results_def_l,
                        roi_mask,
                        conf.ss_size,
                        conf.shape_func,
                        conf.residual_mode);
    save_residual_map(
        residual_map,
        residual_filename(saveconf.basepath, saveconf.prefix,
                          left_image_filename),
        conf.px_hori,
        conf.px_vert,
        saveconf.delimiter,
        saveconf.precision);

    if (!conf.stereo) {
        return;
    }

    if (interp_ref_r == nullptr || interp_def_r == nullptr ||
        results_def_r == nullptr || results_ref_r == nullptr) {
        throw std::invalid_argument(
            "Stereo residual output requires both right-camera images and results");
    }

    ResultArrays right_temporal = *results_def_r;
    right_temporal.p.clear();
    right_temporal.num_params = 0;
    for (int i = 0; i < right_temporal.num_ss; ++i) {
        right_temporal.u[i] = results_def_l.u[i] + results_def_r->u[i] - results_ref_r->u[i];
        right_temporal.v[i] = results_def_l.v[i] + results_def_r->v[i] - results_ref_r->v[i];
    }

    create_residual_map(residual_map,
                        *interp_ref_r,
                        *interp_def_r,
                        subset_grid,
                        right_temporal,
                        roi_mask,
                        conf.ss_size,
                        conf.shape_func,
                        conf.residual_mode);
    save_residual_map(
        residual_map,
        residual_filename(saveconf.basepath, saveconf.prefix,
                          right_image_filename, 1),
        conf.px_hori,
        conf.px_vert,
        saveconf.delimiter,
        saveconf.precision);
}

void save_residual_map(const std::vector<double> &residual_map,
                       const std::string &filename,
                       int width,
                       int height,
                       const std::string &delimiter,
                       int precision) {

    const std::size_t expected = static_cast<std::size_t>(width) * height;

    if (residual_map.size() != expected) {
        throw std::invalid_argument("Residual map dimensions do not match the output image dimensions");
    }

    std::ofstream output(filename);
    if (!output) {
        throw std::runtime_error("Failed to open residual output file: " + filename);
    }

    common_util::CsvBuffer csv(output, delimiter, precision);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            csv.field(residual_map[static_cast<std::size_t>(y) * width + x], x == width - 1);
        }
        csv.newline();
    }
}

std::string residual_filename(const std::string &basepath,
                              const std::string &prefix,
                              const std::string &image_filename,
                              int camera) {
    const std::filesystem::path image_path(image_filename);
    std::string stem = image_path.stem().string();
    if (camera >= 0) {
        stem += "_cam" + std::to_string(camera);
    }
    return (std::filesystem::path(basepath) /
            (prefix + stem + "_residual.csv")).string();
}
