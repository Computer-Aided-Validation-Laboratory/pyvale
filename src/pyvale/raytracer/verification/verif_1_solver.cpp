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
#include <cmath>
#include <iomanip>
#include <format>

#include "../../common_cpp/Eigen/Dense"

#include "cpp/rteigentypes.h"
#include "cpp/rtshapefuncs.h"
#include "cpp/rtrayintersection_extracted.h"
#include "cpp/rtelemconstants.h"

#include "verif_cpp/meshio.h"
#include "verif_cpp/verif.h"
#include "verif_cpp/verifconstants.h"
#include "verif_cpp/orchestration.h"

const std::string verif_subdir_name = "verif_1";

// const std::size_t structured_grid_quad_num = 250;
// const std::size_t structured_grid_tri_num = 250;

// const std::size_t structured_grid_quad_num = 20;
// const std::size_t structured_grid_tri_num = 20;

const std::size_t structured_grid_quad_num = 60;
const std::size_t structured_grid_tri_num = 60;


double directionError(const EiVector3d& d_true,
                const EiVector3d& d_rec)
{
    EiVector3d v1 = d_true.normalized();
    EiVector3d v2 = d_rec.normalized();

    double dot = v1.dot(v2);
    dot = std::max(-1.0, std::min(1.0, dot));
    double error = std::acos(dot) * 180.0 / M_PI;

    if (dot > 1.0 || dot < -1.0)
    {
        std::cout << "Out of range: " << std::setprecision(17)
                  << dot << '\n';
    }

    // std::cout << std::string(80, '-') << " \n";
    // std::cout << "d_true = " << d_true << " \n";
    // std::cout << "d_rec = " << d_rec << " \n";
    // std::cout << "err_dir = " << std::setprecision(17) << error << " deg" <<  " \n";
    // std::cout << std::string(80, '-') << " \n";

    return error;
}

struct SampleGrid
{
    std::vector<SamplePoint> list;
    std::size_t rows_num;
    std::size_t cols_num;
};

template<ElementNodeCount nodes_per_element>
SampleGrid buildSampleList()
{
    std::vector<SamplePoint> sampleList;

    const std::size_t structuredGridNum =
        (nodes_per_element == ElementNodeCount::TRI3 || nodes_per_element == ElementNodeCount::TRI6)
            ? structured_grid_tri_num
            : structured_grid_quad_num;

    GridDims mapDims =
            appendStructuredSamples<nodes_per_element>(
                sampleList,
                structuredGridNum);

    return {
        std::move(sampleList),
        mapDims.rows_num,
        mapDims.cols_num
    };
}

template<ElementNodeCount nodes_per_element>
double forwardMap(
                const double xi, 
                const double eta,
                const EiVector3d& camera_center,
                const NodeArray<nodes_per_element>& nodes,
                EiVector3d& d)
{

    Eigen::VectorXd N(nodes_per_element);
    if constexpr (nodes_per_element == 6) {
        N = compute_shape_tri6(xi, eta);
    }

    EiVector3d P = EiVector3d::Zero();
    for (int i = 0; i < nodes_per_element; ++i)
        P += N[i] * nodes[i];

    d = (P - camera_center).normalized();
    double t = (P - camera_center).norm();

    return t;
}

template<ElementNodeCount nodes_per_element>
double inverseMap(
                const Ray ray,
                const NodeArray<nodes_per_element>& nodes,
                double& xi, 
                double& eta)
{
    EiVector3d n_tmp;
    Eigen::Vector2d gh_rec;
    double t = 0;
    if constexpr (nodes_per_element == 6) {
        t = intersect_tri6(ray, nodes, n_tmp, gh_rec);
    }
    xi = gh_rec.x();
    eta = gh_rec.y();

    return t;
}

template<ElementNodeCount nodes_per_element>
void evalSample(const EiVector3d& camera_center,
                const NodeArray<nodes_per_element>& nodes,
                const SamplePoint& sample,
                SampleRecord& record)
{

    // 2. Calculate true surface location (forward mapping)
    EiVector3d d_true;
    double t_true = forwardMap<nodes_per_element>(
                sample.xi_true, sample.eta_true,
                camera_center,
                nodes,
                d_true);

    Ray ray(camera_center, d_true);

    // 3. Recover ground truth sample
    double xi_rec, eta_rec;
    double t_rec = inverseMap<nodes_per_element>(
                ray,
                nodes,
                xi_rec, eta_rec);
    std::size_t converged = std::isfinite(t_rec) ? 1 : 0;
    bool in_domain = isInParametricDomain<nodes_per_element>(xi_rec, eta_rec);
            
    // 4. Forward map the recovered sample
    EiVector3d d_rec;
    double t_reproj = forwardMap<nodes_per_element>(
                xi_rec, eta_rec,
                camera_center,
                nodes,
                d_rec);

    // 5. Calculate projection error
    double err_xi = xi_rec - sample.xi_true;
    double err_eta = eta_rec - sample.eta_true;
    double err_param = std::sqrt(err_xi * err_xi + err_eta * err_eta);

    double err_dir = directionError(d_true, d_rec);
    double err_t = t_rec - t_true;
    double err_t_reproj = t_reproj - t_true;
    
    // std::cout << "t_true = " << t_true << "\n";
    // std::cout << "t_rec = " << t_rec << "\n";
    // std::cout << "Complete! :) " <<" \n";

    // Load results into sample record
    record.xi_true = sample.xi_true;
    record.eta_true = sample.eta_true;
    record.xi_rec = xi_rec;
    record.eta_rec = eta_rec;
    record.err_xi = err_xi;
    record.err_eta = err_eta;
    record.err_param = err_param;
    record.err_dir = err_dir;
    record.t_true = t_true;
    record.t_rec = t_rec;
    record.t_reproj = t_reproj;
    record.err_t = err_t;
    record.err_t_reproj = err_t_reproj;
    record.row_idx = sample.row_idx;
    record.col_idx = sample.col_idx;
    record.converged = converged;
    record.in_domain = static_cast<std::size_t>(in_domain);
}


void saveFieldMaps(
    const std::string& out_dir,
    size_t frame_idx,
    size_t rows_num,
    size_t cols_num,
    const std::vector<double>& field0,
    const std::vector<double>& field1
) {

    std::string field0_stem =
        "frame_" + std::to_string(frame_idx) + "_field0";
    std::string field0_csv = field0_stem + ".csv";

    writeScalarMapCsv(
        out_dir + "/" + field0_csv,
        rows_num,
        cols_num,
        field0
    );

    writeScalarMapBmp(
        out_dir + "/" + field0_stem + ".bmp",
        field0,
        rows_num,
        cols_num
    );

    std::string field1_stem =
        "frame_" + std::to_string(frame_idx) + "_field1";
    std::string field1_csv = field1_stem + ".csv";

    writeScalarMapCsv(
        out_dir + "/" + field1_csv,
        rows_num,
        cols_num,
        field1
    );

    writeScalarMapBmp(
        out_dir + "/" + field1_stem + ".bmp",
        field1,
        rows_num,
        cols_num
    );
}


template<ElementNodeCount nodes_per_element>
NodeArray<nodes_per_element> frameNodes(
    const SimData& sim_data,
    size_t frame_idx
) {
    NodeArray<nodes_per_element> nodes;

    const Eigen::Map<const Eigen::VectorXi>& elem = 
        sim_data.connect.elem(0);

    for (size_t nn = 0; nn < nodes_per_element; ++nn) {
        size_t node_idx = elem[nn];

        nodes[nn] = sim_data.coords.mat.row(node_idx);

        if (sim_data.field) {
            const Field& field = *sim_data.field;

            nodes[nn](0) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx),
                                     0);
            
            nodes[nn](1) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx),
                                     1);
            
            nodes[nn](2) += field.at(static_cast<int>(frame_idx),
                                     static_cast<int>(node_idx),
                                     2);
        }
    }

    return nodes;
}


const char* elementNodeCountToString(ElementNodeCount nodes_per_element)
{
    switch (nodes_per_element)
    {
        case TRI3:  return "tri3";
        case TRI6:  return "tri6";
        case QUAD4: return "quad4";
        case QUAD8: return "quad8";
        case QUAD9: return "quad9";
        default:    return "unknown";
    }
}

template<ElementNodeCount nodes_per_element>
void runDistortCase(
    const DistortCase& case_spec,
    std::vector<double>& global_reproj_errs)
{

    // std::string out_dir_path =
    //     output_dir_name + "/" +
    //     verif_subdir_name + "/a_distort_" +
    //     case_spec.case_name + "_" +
    //     std::to_string(static_cast<int>(nodes_per_element));

    std::string out_dir_path =
        output_dir_name + "/" +
        verif_subdir_name + "/a_distort_" +
        case_spec.case_name + "_" +
        elementNodeCountToString(nodes_per_element);

    SimData sim_data = loadData(case_spec.data_dir);
    // NodeArray<nodes_per_element> nodes =
    //     getNodes<nodes_per_element>(sim_data.coords, sim_data.connect, 0);
    SampleGrid sample_data = buildSampleList<nodes_per_element>();

    size_t time_steps = sim_data.field
        ? static_cast<size_t>(sim_data.field->getTimeN())
        : 1;
    const std::size_t map_len =
        sample_data.rows_num * sample_data.cols_num;

    // Iterate over all frames
    for (size_t frame_idx = 0; frame_idx < time_steps; ++frame_idx)
    {
        NodeArray<nodes_per_element> nodes =
            frameNodes<nodes_per_element>(sim_data, frame_idx);

        const std::size_t sample_count = sample_data.list.size();
        std::vector<SampleRecord> records(sample_count);

        std::vector<double> xi_map(map_len);
        std::vector<double> eta_map(map_len);
        std::vector<double> reproj_vals;
        std::vector<double> param_vals;

        // Iterate over all sampling locations for one element configuration
        #pragma omp parallel
        {
            std::vector<double> local_reproj_vals;
            std::vector<double> local_param_vals;
            std::vector<double> local_global_reproj;

            #pragma omp for
            for (std::ptrdiff_t i = 0;
                 i < static_cast<std::ptrdiff_t>(sample_count);
                 ++i)
            {
                const SamplePoint& sample = sample_data.list[i];

                SampleRecord record;
                evalSample<nodes_per_element>(
                    case_spec.camera_center,
                    nodes, sample,
                    record);

                records[i] = std::move(record);

                const std::size_t map_idx =
                    sample.row_idx * sample_data.cols_num +
                    sample.col_idx;
                xi_map[map_idx] = records[i].xi_rec;
                eta_map[map_idx] = records[i].eta_rec;

                if (std::isfinite(records[i].err_t_reproj))
                {
                    local_reproj_vals.push_back(records[i].err_t_reproj);
                    local_global_reproj.push_back(records[i].err_t_reproj);
                }

                if (std::isfinite(records[i].err_param))
                {
                    local_param_vals.push_back(records[i].err_param);
                }
            }

            #pragma omp critical
            {
                reproj_vals.insert(
                    reproj_vals.end(),
                    local_reproj_vals.begin(),
                    local_reproj_vals.end());

                param_vals.insert(
                    param_vals.end(),
                    local_param_vals.begin(),
                    local_param_vals.end());

                global_reproj_errs.insert(
                    global_reproj_errs.end(),
                    local_global_reproj.begin(),
                    local_global_reproj.end());
            }
        }

        std::string stats_file_name =
            "solver_stats_frame" +
            std::to_string(frame_idx) +
            ".csv";

        writeSolverStatsCsv(
            out_dir_path,
            stats_file_name,
            records);

        saveFieldMaps(
            out_dir_path,
            frame_idx,
            sample_data.rows_num,
            sample_data.cols_num,
            xi_map,
            eta_map);
            

        if (!reproj_vals.empty() && !param_vals.empty())
        {
            auto reproj_stats = calcScalarStats(reproj_vals);
            auto param_stats = calcScalarStats(param_vals);

            std::cout
                << "a_distort_" << case_spec.case_name << "_"
                << nodes_per_element
                << " frame " << frame_idx
                << ": reproj max=" << reproj_stats.max
                << " param max=" << param_stats.max
                << "\n";
        }
        else
        {
            std::cout
                << "a_distort_" << case_spec.case_name << "_"
                << nodes_per_element
                << " frame " << frame_idx
                << ": no converged samples\n";
        }
    }
    std::cout << std::string(80, '-') << " \n";

    // std::cout << "out_dir_path = " << out_dir_path << "\n";
    // std::cout << "solver_stats_frame = " << stats_file_name << "\n";
    
    // std::cout << "Nodes per element = " << static_cast<int>(nodes_per_element) << "\n";
};

int main(){

    std::vector<double> global_reproj_errs;

    for (const DistortCase& case_spec : distort_cases) {
        std::cout << case_spec.case_name << case_spec.data_dir << "\n";
        switch (case_spec.nodes_per_element) {
            // case ElementNodeCount::TRI3:
            //     runDistortCase<ElementNodeCount::TRI3>(
            //         case_spec);
            //     break;
        
            case ElementNodeCount::TRI6:
                runDistortCase<ElementNodeCount::TRI6>(
                    case_spec,
                    global_reproj_errs);
                break;
    
        }
    }

    std::cout << "Complete! :) " <<" \n";
    return 0;

}
