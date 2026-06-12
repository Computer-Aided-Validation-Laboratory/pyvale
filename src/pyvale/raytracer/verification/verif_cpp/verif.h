#pragma once

#include <iostream>
#include <limits>
#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cassert>
#include <memory>
#include <optional>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <cstdint>
#include <algorithm>

#include "../../../common_cpp/Eigen/Dense"

#include "../cpp/rtelemconstants.h"

const std::string output_dir_name = "verif";

constexpr std::size_t INVALID_GRID_IDX =
    static_cast<std::size_t>(-1);

struct SamplePoint
{
    double xi_true;
    double eta_true;

    std::size_t row_idx = INVALID_GRID_IDX;
    std::size_t col_idx = INVALID_GRID_IDX;
};

struct GridDims
{
    std::size_t rows_num;
    std::size_t cols_num;
};

struct SampleRecord
{
    double xi_true;
    double eta_true;
    double xi_rec;
    double eta_rec;
    double err_xi;
    double err_eta;
    double err_param;
    double err_dir;
    double t_true;
    double t_rec;
    double t_reproj;
    double err_t;
    double err_t_reproj;
    std::size_t converged; 
    std::size_t in_domain;

    std::size_t row_idx = INVALID_GRID_IDX;
    std::size_t col_idx = INVALID_GRID_IDX;
};

struct ScalarStats
{
    double min;
    double q1;
    double median;
    double q3;
    double max;
    double mean;
    double rms;
};

template<ElementNodeCount nodes_per_element>
GridDims appendStructuredSamples(
    std::vector<SamplePoint>& samples,
    std::size_t gridNum)
{
    if constexpr (
        nodes_per_element == ElementNodeCount::TRI3 ||
        nodes_per_element == ElementNodeCount::TRI6)
    {
        for (std::size_t rr = 0; rr < gridNum; ++rr)
        {
            double eta =
                static_cast<double>(rr) /
                static_cast<double>(gridNum - 1);

            for (std::size_t cc = 0; cc < gridNum; ++cc)
            {
                double xi =
                    static_cast<double>(cc) /
                    static_cast<double>(gridNum - 1);

                if (xi + eta <= 1.0)
                {
                    samples.push_back({xi, eta, rr, cc});
                }
            }
        }
    }
    else
    {
        double xi_min = -1.0;
        double xi_max = 1.0;

        double eta_min = -1.0;
        double eta_max = 1.0;

        for (std::size_t rr = 0; rr < gridNum; ++rr)
        {
            double eta =
                eta_min +
                (static_cast<double>(rr) /
                 static_cast<double>(gridNum - 1)) *
                (eta_max - eta_min);

            for (std::size_t cc = 0; cc < gridNum; ++cc)
            {
                double xi =
                    xi_min +
                    (static_cast<double>(cc) /
                     static_cast<double>(gridNum - 1)) *
                    (xi_max - xi_min);

                samples.push_back({xi, eta, rr, cc});
            }
        }
    }

    return {gridNum, gridNum};
};

void writeSolverStatsCsv(
    const std::string& out_dir,
    const std::string& file_name,
    const std::vector<SampleRecord>& records)
{

    std::filesystem::path csv_path =
        std::filesystem::path(out_dir) / file_name;
    std::filesystem::create_directories(csv_path.parent_path());

    std::ofstream file(csv_path);

    if (!file)
    {
        throw std::system_error(
            errno,
            std::generic_category(),
            "Failed to open file: " + csv_path.string());
    }

    
    file << std::setprecision(17);
    file
        << "ideal_xi,ideal_eta,"
        << "solved_xi,solved_eta,"
        << "err_xi,err_eta,"
        << "err_param,err_dir,"
        << "t_true,t_rec,t_reproj,"
        << "err_t,reproj_err,"
        << "converged,"
        << "in_domain,"
        << "row_idx,col_idx\n";

    for (const SampleRecord& record : records)
    {
        file
            << record.xi_true << ',' << record.eta_true << ','
            << record.xi_rec << ',' << record.eta_rec << ','
            << record.err_xi << ',' << record.err_eta << ','
            << record.err_param << ',' << record.err_dir << ','
            << record.t_true << ',' << record.t_rec << ',' << record.t_reproj << ','
            << record.err_t << ',' << record.err_t_reproj << ','
            << record.converged << ','
            << record.in_domain << ','
            << record.row_idx << ',' << record.col_idx
            << '\n';
    }

    if (!file.good())
    {
        throw std::runtime_error(
            "Failed while writing file: " + csv_path.string());
    }
};


void writeScalarMapCsv(
    const std::string& file_name,
    size_t rows_num,
    size_t cols_num,
    const std::vector<double>& vals
) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_name);
    }

    for (size_t rr = 0; rr < rows_num; ++rr) {
        for (size_t cc = 0; cc < cols_num; ++cc) {
            file << vals[rr * cols_num + cc];

            if (cc + 1 < cols_num)
                file << ",";
        }
        file << "\n";
    }
};


#pragma pack(push, 1)
struct BMPHeader {
    uint16_t bfType = 0x4D42; // "BM"
    uint32_t bfSize = 0;
    uint32_t bfReserved = 0;
    uint32_t bfOffBits = 54 + 256 * 4; // header + palette

    uint32_t biSize = 40;
    int32_t  biWidth = 0;
    int32_t  biHeight = 0;
    uint16_t biPlanes = 1;
    uint16_t biBitCount = 8;
    uint32_t biCompression = 0;
    uint32_t biSizeImage = 0;
    int32_t  biXPelsPerMeter = 0;
    int32_t  biYPelsPerMeter = 0;
    uint32_t biClrUsed = 256;
    uint32_t biClrImportant = 256;
};
#pragma pack(pop)

static void writeScalarMapBmp(
    const std::string& filename,
    const std::vector<double>& vals,
    size_t rows,
    size_t cols) 
{
    // Get min/max for scaling
    auto [min_it, max_it] = std::minmax_element(vals.begin(), vals.end());
    double vmin = *min_it;
    double vmax = *max_it;
    double range = (vmax - vmin);
    if (range == 0.0) range = 1.0;

    // Pad rows to 4 bytes
    size_t rowSize = (cols + 3) & ~3;
    size_t imageSize = rowSize * rows;

    BMPHeader hdr;
    hdr.biWidth = static_cast<int32_t>(cols);
    hdr.biHeight = static_cast<int32_t>(rows);
    hdr.bfSize = hdr.bfOffBits + imageSize;
    hdr.biSizeImage = imageSize;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open BMP file: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));

    // Grayscale
    for (int i = 0; i < 256; ++i) {
        uint8_t c = static_cast<uint8_t>(i);
        file.put(c);
        file.put(c);
        file.put(c);
        file.put(0);
    }

    // Write pixel data
    std::vector<uint8_t> row(rowSize, 0);

    for (size_t r = 0; r < rows; ++r) {
        size_t rr = rows - 1 - r; // bottom-up

        for (size_t c = 0; c < cols; ++c) {
            double v = vals[rr * cols + c];
            row[c] = static_cast<uint8_t>(255.0 * (v - vmin) / range);
        }

        file.write(reinterpret_cast<const char*>(row.data()), rowSize);
    }
}

double quantileFromSorted(const std::vector<double>& sorted_vals, double q) {
    size_t n = sorted_vals.size();
    double pos = q * (n - 1);
    size_t i = static_cast<size_t>(pos);
    double frac = pos - i;

    if (i + 1 < n) {
        return sorted_vals[i] * (1.0 - frac) + sorted_vals[i + 1] * frac;
    }
    return sorted_vals[i];
}

ScalarStats calcScalarStats(const std::vector<double>& vals) {
    std::vector<double> sorted_vals = vals; // copy

    std::sort(sorted_vals.begin(), sorted_vals.end());

    double sum = 0.0;
    double sum_sq = 0.0;

    for (double v : vals) {
        sum += v;
        sum_sq += v * v;
    }

    double inv_len = 1.0 / static_cast<double>(vals.size());

    ScalarStats stats;
    stats.min    = sorted_vals.front();
    stats.q1     = quantileFromSorted(sorted_vals, 0.25);
    stats.median = quantileFromSorted(sorted_vals, 0.5);
    stats.q3     = quantileFromSorted(sorted_vals, 0.75);
    stats.max    = sorted_vals.back();
    stats.mean   = sum * inv_len;
    stats.rms    = std::sqrt(sum_sq * inv_len);

    return stats;
};


template<ElementNodeCount nodes_per_element>
bool isInParametricDomain(double xi, double eta)
{
    const double eps = 1.0e-8;

    if constexpr (nodes_per_element == ElementNodeCount::TRI3 ||
                  nodes_per_element == ElementNodeCount::TRI6)
    {
        return xi >= -eps &&
               eta >= -eps &&
               xi + eta <= 1.0 + eps;
    }
    else
    {
        return std::abs(xi) <= 1.0 + eps &&
               std::abs(eta) <= 1.0 + eps;
    }
}
