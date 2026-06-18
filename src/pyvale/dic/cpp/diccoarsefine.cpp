// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

#include "./dicfourier.hpp"
#include "./diccoarsefine.hpp"

namespace coarsefine {

    // Downsample an image by factor 2 using simple averaging
    std::vector<double> downsample(const double* img, int width, int height) {
        int new_w = width / 2;
        int new_h = height / 2;
        std::vector<double> out(new_w * new_h, 0.0);
        for (int y = 0; y < new_h; ++y) {
            for (int x = 0; x < new_w; ++x) {
                out[y * new_w + x] = 0.25 * (
                    img[(2*y)   * width + (2*x)  ] +
                    img[(2*y)   * width + (2*x+1)] +
                    img[(2*y+1) * width + (2*x)  ] +
                    img[(2*y+1) * width + (2*x+1)]
                );
            }
        }
        return out;
    }

    // Extract a subset patch centered at (cx, cy) from an image of given width
    // Returns false if the patch would go out of bounds
    bool extract_subset(const double* img, int img_w, int img_h,
                        int cx, int cy, int ss_size,
                        std::vector<double>& patch) {
        int half = ss_size / 2;
        int x0 = cx - half, y0 = cy - half;
        if (x0 < 0 || y0 < 0 || x0 + ss_size > img_w || y0 + ss_size > img_h)
            return false;
        patch.resize(ss_size * ss_size);
        for (int y = 0; y < ss_size; ++y)
            for (int x = 0; x < ss_size; ++x)
                patch[y * ss_size + x] = img[(y0 + y) * img_w + (x0 + x)];
        return true;
    }

    CoarseToFineResult coarse_to_fine_search(
        const double* img_ref,
        const double* img_def,
        int img_w,
        int img_h,
        int center_x,         // subset center in the reference image
        int center_y,
        int ss_size,          // subset size at full resolution (e.g. 51)
        int max_displacement, // max displacement at full resolution (e.g. 1000)
        bool subpx,
        const std::string& peak_method
    ) {
        // --- Build image pyramids ---
        // Number of levels: enough to reduce max_displacement to ~1 pixel
        int n_levels = static_cast<int>(std::ceil(std::log2(max_displacement))) + 1;

        // Clamp so the image doesn't shrink below the subset size
        while (n_levels > 1) {
            int scale = 1 << (n_levels - 1);
            if ((img_w / scale) < ss_size || (img_h / scale) < ss_size)
                --n_levels;
            else
                break;
        }

        // Build pyramid: level 0 = original, level n-1 = coarsest
        std::vector<std::vector<double>> pyr_ref(n_levels);
        std::vector<std::vector<double>> pyr_def(n_levels);
        std::vector<int> pyr_w(n_levels), pyr_h(n_levels);

        pyr_w[0] = img_w; pyr_h[0] = img_h;
        // level 0 points into original data (we copy to vector for uniformity)
        pyr_ref[0].assign(img_ref, img_ref + img_w * img_h);
        pyr_def[0].assign(img_def, img_def + img_w * img_h);

        for (int lv = 1; lv < n_levels; ++lv) {
            pyr_w[lv] = pyr_w[lv-1] / 2;
            pyr_h[lv] = pyr_h[lv-1] / 2;
            pyr_ref[lv] = downsample(pyr_ref[lv-1].data(), pyr_w[lv-1], pyr_h[lv-1]);
            pyr_def[lv] = downsample(pyr_def[lv-1].data(), pyr_w[lv-1], pyr_h[lv-1]);
        }

        // --- Coarse-to-fine refinement ---
        double est_x = 0.0, est_y = 0.0; // displacement estimate (in full-res pixels)
        CoarseToFineResult result;

        for (int lv = n_levels - 1; lv >= 0; --lv) {
            double scale = static_cast<double>(1 << lv); // full-res pixels per coarse pixel

            // Map full-res center and current estimate to this level
            int lv_cx = static_cast<int>(std::round(center_x / scale));
            int lv_cy = static_cast<int>(std::round(center_y / scale));

            // Predicted deformed center at this level
            double pred_dx = est_x / scale;
            double pred_dy = est_y / scale;
            int lv_def_cx = static_cast<int>(std::round(lv_cx + pred_dx));
            int lv_def_cy = static_cast<int>(std::round(lv_cy + pred_dy));

            // Clamp to image bounds (with half-subset margin)
            int half = ss_size / 2;
            lv_def_cx = std::clamp(lv_def_cx, half, pyr_w[lv] - half - 1);
            lv_def_cy = std::clamp(lv_def_cy, half, pyr_h[lv] - half - 1);
            lv_cx     = std::clamp(lv_cx,     half, pyr_w[lv] - half - 1);
            lv_cy     = std::clamp(lv_cy,     half, pyr_h[lv] - half - 1);

            // Extract subsets
            std::vector<double> patch_ref, patch_def;
            bool ok_ref = extract_subset(pyr_ref[lv].data(), pyr_w[lv], pyr_h[lv],
                                        lv_cx, lv_cy, ss_size, patch_ref);
            bool ok_def = extract_subset(pyr_def[lv].data(), pyr_w[lv], pyr_h[lv],
                                        lv_def_cx, lv_def_cy, ss_size, patch_def);
            if (!ok_ref || !ok_def) continue;

            // Run FFT cross-correlation
            FFT fft(ss_size, ss_size, true);
            fft.ss_ref.vals = patch_ref;
            fft.ss_def.vals = patch_def;
            // fft.zero_norm_subset(fft.ss_ref, ss_size, ss_size);
            // fft.zero_norm_subset(fft.ss_def, ss_size, ss_size);
            fft.correlate();

            std::cout << std::endl;
            for (int row = 0; row < ss_size; ++row) {
                for (int col = 0; col < ss_size; ++col) {
                    int idx  = row*ss_size+col;
                    std::cout << col << " " << row << " ";
                    std::cout << fft.ss_ref.x[idx] << " " << fft.ss_ref.y[idx] << " " << fft.ss_ref.vals[idx] << " ";
                    std::cout << fft.ss_def.x[idx] << " " << fft.ss_def.y[idx] << " " << fft.ss_def.vals[idx] << " ";
                    std::cout << fft.cross_corr[idx] << std::endl;
                }
            }

            // Get peak — use get_peak_offset which centers the result for you
            double px, py, pval;
            // Only use subpixel at the finest level to avoid over-refining coarse estimates
            bool do_subpx = subpx && (lv == 0);
            // fft.get_peak_offset(px, py, pval, do_subpx, peak_method);
            // std::cout << px << " " << py << std::endl;
            fft.get_peak(px, py, pval, do_subpx, peak_method);

            //std::cout << px << " " << py << std::endl;

            // Residual displacement at this level, scaled back to full-res pixels
            double residual_x = px * scale;
            double residual_y = py * scale;

            est_x += residual_x;
            est_y += residual_y;

            if (lv == 0) {
                result.disp_x  = est_x;
                result.disp_y  = est_y;
                result.peak_val = pval;
                result.success  = true;
            }
        }

        return result;
    }

}
