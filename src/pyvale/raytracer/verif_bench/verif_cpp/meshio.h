#pragma once

#include <array>
#include <string>
#include <vector>
#include <fstream>
#include <optional>
#include <cstdlib>

#include "../../../common_cpp/Eigen/Dense"

#include "verifconstants.h"

struct Coords {
    MatXf mat; // N x 3

    explicit Coords(int n = 0)
        : mat(n, 3) {}

    inline int size() const { return (int)mat.rows(); }

    inline double x(int i) const { return mat(i, 0); }
    inline double y(int i) const { return mat(i, 1); }
    inline double z(int i) const { return mat(i, 2); }

    inline Vec3f vec(int i) const {
        return mat.row(i);
    }
};

struct Connect {
    MatXi table; // E x nodesPerElem

    Connect() = default;

    Connect(int elems, int nodesPerElem)
        : table(elems, nodesPerElem) {}

    inline int elems() const { return (int)table.rows(); }
    inline int nodesPerElem() const { return (int)table.cols(); }

    inline Eigen::Map<const Eigen::VectorXi> elem(int e) const {
        return Eigen::Map<const Eigen::VectorXi>(table.row(e).data(), table.cols());
    }
};

struct Field {
    int timeN = 0;
    int coordN = 0;
    int fieldN = 0;

    std::vector<double> data; // T * C * F

    Field() = default;

    Field(int t, int c, int f)
        : timeN(t), coordN(c), fieldN(f),
          data((size_t)t * c * f, 0.0) {}

    inline int getTimeN() const {
        return timeN;
    }

    inline double& at(int t, int c, int f) {
        return data[((t * coordN) + c) * fieldN + f];
    }

    inline const double& at(int t, int c, int f) const {
        return data[((t * coordN) + c) * fieldN + f];
    }

    // View (T*C) x F
    inline Eigen::Map<MatXf> asMatrix() {
        return Eigen::Map<MatXf>(data.data(), timeN * coordN, fieldN);
    }
};

static inline std::vector<std::string_view> splitCSV(std::string_view line) {
    std::vector<std::string_view> out;

    const char* start = line.data();
    const char* end = start + line.size();

    const char* p = start;

    while (p <= end) {
        const char* q = p;
        while (q < end && *q != ',') q++;

        out.emplace_back(p, q - p);

        p = q + 1;
        if (q >= end) break;
    }

    return out;
}

static inline double fastToDouble(std::string_view s) {
    return std::strtod(std::string(s).c_str(), nullptr);
}

static inline int fastToInt(std::string_view s) {
    return std::atoi(std::string(s).c_str());
}

static Coords parseCoords(const std::vector<std::string>& lines) {
    Coords coords((int)lines.size());

    for (int i = 0; i < (int)lines.size(); i++) {
        auto cols = splitCSV(lines[i]);

        coords.mat(i, 0) = fastToDouble(cols[0]);
        coords.mat(i, 1) = fastToDouble(cols[1]);
        coords.mat(i, 2) = fastToDouble(cols[2]);
    }

    return coords;
}

static Connect parseConnect(const std::vector<std::string>& lines) {
    auto first = splitCSV(lines[0]);
    int nodesPerElem = (int)first.size();

    Connect conn((int)lines.size(), nodesPerElem);

    for (int e = 0; e < (int)lines.size(); e++) {
        auto cols = splitCSV(lines[e]);

        for (int n = 0; n < nodesPerElem; n++) {
            conn.table(e, n) = fastToInt(cols[n]);
        }
    }

    return conn;
}

static int getTimeN(const std::string& line) {
    return (int)splitCSV(line).size();
}

static void parseField(
    const std::vector<std::string>& lines,
    Field& field,
    int fieldIdx
) {
    for (int c = 0; c < (int)lines.size(); c++) {
        auto cols = splitCSV(lines[c]);

        for (int t = 0; t < (int)cols.size(); t++) {
            field.at(t, c, fieldIdx) = fastToDouble(cols[t]);
        }
    }
}

struct SimData {
    Coords coords;
    Connect connect;
    std::optional<Field> field;
    std::optional<Field> disp;
};

static std::vector<std::string> readLines(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> out;
    std::string line;

    while (std::getline(file, line)) {
        if (!line.empty())
            out.emplace_back(std::move(line));
    }

    return out;
}

static SimData loadSimData(
    const std::string& coordPath,
    const std::string& connectPath,
    const std::vector<std::string>* fieldPaths = nullptr,
    const std::vector<std::string>* dispPaths = nullptr) 
{

    auto coordLines = readLines(coordPath);
    Coords coords = parseCoords(coordLines);

    auto connLines = readLines(connectPath);
    Connect conn = parseConnect(connLines);

    std::optional<Field> field;
    if (fieldPaths && !fieldPaths->empty()) {
        auto lines = readLines((*fieldPaths)[0]);

        int timeN = getTimeN(lines[0]);
        int coordN = (int)lines.size();
        int fieldN = (int)fieldPaths->size();

        field.emplace(timeN, coordN, fieldN);
        parseField(lines, *field, 0);

        for (int i = 1; i < fieldPaths->size(); i++) {
            lines = readLines((*fieldPaths)[i]);
            parseField(lines, *field, i);
        }
    }

    std::optional<Field> disp;
    if (dispPaths && !dispPaths->empty()) {
        auto lines = readLines((*dispPaths)[0]);

        int timeN = getTimeN(lines[0]);
        int coordN = (int)lines.size();
        int fieldN = (int)dispPaths->size();

        disp.emplace(timeN, coordN, fieldN);
        parseField(lines, *disp, 0);

        for (int i = 1; i < dispPaths->size(); i++) {
            lines = readLines((*dispPaths)[i]);
            parseField(lines, *disp, i);
        }
    }

    return {
        std::move(coords),
        std::move(conn),
        std::move(field),
        std::move(disp)
    };
}

std::array<EiVector3d, ElementNodeCount::TRI6>
getTri6Nodes(
    const Coords& coords,
    const Connect& connect,
    int elemIdx)
{
    std::array<EiVector3d, ElementNodeCount::TRI6> nodes;

    for (int localNode = 0; localNode < ElementNodeCount::TRI6; ++localNode)
    {
        const int globalNode =
            connect.table(elemIdx, localNode);

        nodes[localNode] =
            coords.mat.row(globalNode);
    }

    return nodes;
}


template<ElementNodeCount nodes_per_element>
NodeArray<nodes_per_element>
getNodes(
    const Coords& coords,
    const Connect& connect,
    const int elemIdx)
{
    NodeArray<nodes_per_element> nodes;

    for (size_t localNode = 0; localNode < static_cast<size_t>(nodes_per_element); ++localNode)
    {
        const int globalNode = connect.table(elemIdx, localNode);
        nodes[localNode] = coords.mat.row(globalNode);
    }

    return nodes;
}