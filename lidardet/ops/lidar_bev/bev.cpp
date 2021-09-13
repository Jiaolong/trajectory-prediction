// pybind libraries
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// C/C++ includes
#include <cfloat>
#include <chrono>
#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace std;

/*
 * @brief returns all the voxels that are traversed by a ray going from start to end
 * @param start : continous world position where the ray starts
 * @param end   : continous world position where the ray end
 * @return vector of voxel ids hit by the ray in temporal order
 *
 * J. Amanatides, A. Woo. A Fast Voxel Traversal Algorithm for Ray Tracing. Eurographics '87
 *
 * Code adapted from: https://github.com/francisengelmann/fast_voxel_traversal
 *
 * Warning:
 *   This is not production-level code.
 */
inline void _voxel_traversal(std::vector<Eigen::Vector3i>& visited_voxels,
    const Eigen::Vector3d& ray_start,
    const Eigen::Vector3d& ray_end,
    const Eigen::VectorXf lidar_range, 
    const double voxel_size)
{
    int xmin = floor(lidar_range[0] / voxel_size);
    int xmax = floor(lidar_range[1] / voxel_size);
    int ymin = floor(lidar_range[2] / voxel_size);
    int ymax = floor(lidar_range[3] / voxel_size);
    int zmin = floor(lidar_range[4] / voxel_size);
    int zmax = floor(lidar_range[5] / voxel_size);

    // Compute normalized ray direction.
    Eigen::Vector3d ray = ray_end - ray_start;
    // ray.normalize();

    // This id of the first/current voxel hit by the ray.
    // Using floor (round down) is actually very important,
    // the implicit int-casting will round up for negative numbers.
    Eigen::Vector3i current_voxel(floor(ray_start[0] / voxel_size),
        floor(ray_start[1] / voxel_size),
        floor(ray_start[2] / voxel_size));

    // create aliases for indices of the current voxel
    int &vx = current_voxel[0], &vy = current_voxel[1], &vz = current_voxel[2];

    // The id of the last voxel hit by the ray.
    // TODO: what happens if the end point is on a border?
    Eigen::Vector3i last_voxel(floor(ray_end[0] / voxel_size),
        floor(ray_end[1] / voxel_size),
        floor(ray_end[2] / voxel_size));

    // In which direction the voxel ids are incremented.
    int stepX = (ray[0] >= 0) ? 1 : -1; // correct
    int stepY = (ray[1] >= 0) ? 1 : -1; // correct
    int stepZ = (ray[2] >= 0) ? 1 : -1; // correct

    // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
    double next_voxel_boundary_x = (vx + stepX) * voxel_size; // correct
    double next_voxel_boundary_y = (vy + stepY) * voxel_size; // correct
    double next_voxel_boundary_z = (vz + stepZ) * voxel_size; // correct

    // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
    // the value of t at which the ray crosses the first vertical voxel boundary
    double tMaxX = (ray[0] != 0) ? (next_voxel_boundary_x - ray_start[0]) / ray[0] : DBL_MAX; //
    double tMaxY = (ray[1] != 0) ? (next_voxel_boundary_y - ray_start[1]) / ray[1] : DBL_MAX; //
    double tMaxZ = (ray[2] != 0) ? (next_voxel_boundary_z - ray_start[2]) / ray[2] : DBL_MAX; //

    // tDeltaX, tDeltaY, tDeltaZ --
    // how far along the ray we must move for the horizontal component to equal the width of a voxel
    // the direction in which we traverse the grid
    // can only be FLT_MAX if we never go in that direction
    double tDeltaX = (ray[0] != 0) ? voxel_size / ray[0] * stepX : DBL_MAX;
    double tDeltaY = (ray[1] != 0) ? voxel_size / ray[1] * stepY : DBL_MAX;
    double tDeltaZ = (ray[2] != 0) ? voxel_size / ray[2] * stepZ : DBL_MAX;

    // Note: I am not sure why there is a need to do this, but I am keeping it for now
    // possibly explained by: https://github.com/francisengelmann/fast_voxel_traversal/issues/6
    Eigen::Vector3i diff(0, 0, 0);
    bool neg_ray = false;
    if (vx != last_voxel[0] && ray[0] < 0) {
        diff[0]--;
        neg_ray = true;
    }
    if (vy != last_voxel[1] && ray[1] < 0) {
        diff[1]--;
        neg_ray = true;
    }
    if (vz != last_voxel[2] && ray[2] < 0) {
        diff[2]--;
        neg_ray = true;
    }
    visited_voxels.push_back(current_voxel);
    if (neg_ray) {
        current_voxel += diff;
        visited_voxels.push_back(current_voxel);
    }

    // ray casting loop
    bool truncated = false;
    while (current_voxel != last_voxel) {
        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                vx += stepX;
                truncated = (vx < xmin || vx >= xmax);
                tMaxX += tDeltaX;
            } else {
                vz += stepZ;
                truncated = (vz < zmin || vz >= zmax);
                tMaxZ += tDeltaZ;
            }
        } else {
            if (tMaxY < tMaxZ) {
                vy += stepY;
                truncated = (vy < ymin || vy >= ymax);
                tMaxY += tDeltaY;
            } else {
                vz += stepZ;
                truncated = (vz < zmin || vz >= zmax);
                tMaxZ += tDeltaZ;
            }
        }
        
        if (truncated)
            break;

        visited_voxels.push_back(current_voxel);
    }
}

Eigen::MatrixXf rgb_traversability_map(
    const Eigen::MatrixXf& points,     /* Nx5 ndarray, [x, y, z, intensity, obstacle] */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double sensor_height,              /* lidar mounted height */
    double bev_res /* meters per pixel*/)
{

    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];
    
    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    // bev_map has 5 channels: intensity, height, density, inferenced state, initial state
    // cell state code, 0: unknow, 1: obstacle, 2: free space
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w, 5);

    std::vector<ssize_t> valid_ids;
    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;
        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;
        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;
        
        valid_ids.push_back(n);
        auto i = points(n, 3);
        auto o = points(n, 4);
        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        double h = (z - zmin) / (zmax - zmin);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);

        int idx = py * img_w + px;
        bev_map(idx, 0) += i;
        if (h > bev_map(idx, 1))
            bev_map(idx, 1) = h;
        bev_map(idx, 2) += 1;

        if (o == 1)
            bev_map(idx, 4) = 1;
        else
            bev_map(idx, 4) = 2;

        bev_map(idx, 3) = bev_map(idx, 4);
    }

    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            int idx = py * img_w + px;
            auto c = bev_map(idx, 2);
            if (c > 0) {
                bev_map(idx, 0) /= c;
            }
            bev_map(idx, 2) = std::min(1.0, log(c + 1) / log(64));
        }
    }
    
    Eigen::Vector3d origin(0, 0, sensor_height);

    // compute visibility
    for (auto n : valid_ids) {
        Eigen::Vector3d point = points.row(n).head(3).cast<double>();
        std::vector<Eigen::Vector3i> visited_voxels;
        _voxel_traversal(visited_voxels, origin, point, lidar_range, bev_res);
        const int M = visited_voxels.size();
        for (int j = 0; j < M; ++j) {
            int vx = int(visited_voxels[j][0] - xmin / bev_res);
            int vy = int(ymax / bev_res - visited_voxels[j][1]);
            int vidx = vy * img_w + vx;
            if (bev_map(vidx, 3) != 0)
                continue;
            
            bev_map(vidx, 3) = 2; // free
        } // M
    }
    
    return bev_map;
}

Eigen::MatrixXf rgb_map(
    const Eigen::MatrixXf& points,     /* Nx4 ndarray */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double bev_res /* m per pixel*/)
{
    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];

    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w, 3);

    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;
        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;
        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;

        auto i = points(n, 3);
        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        double h = (z - zmin) / (zmax - zmin);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);

        int idx = py * img_w + px;
        bev_map(idx, 0) += i;
        if (h > bev_map(idx, 1))
            bev_map(idx, 1) = h;
        bev_map(idx, 2) += 1;
    }

    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            int idx = py * img_w + px;
            auto c = bev_map(idx, 2);
            if (c > 0) {
                bev_map(idx, 0) /= c;
            }
            bev_map(idx, 2) = std::min(1.0, log(c + 1) / log(64));
        }
    }
    return bev_map;
}

Eigen::MatrixXf height_map(
    const Eigen::MatrixXf& points,     /* Nx4 ndarray */
    const Eigen::VectorXf lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
    double bev_res /* m per pixel*/)
{
    auto xmin = lidar_range[0];
    auto xmax = lidar_range[1];
    auto ymin = lidar_range[2];
    auto ymax = lidar_range[3];
    auto zmin = lidar_range[4];
    auto zmax = lidar_range[5];

    int img_w = int((xmax - xmin) / bev_res);
    int img_h = int((ymax - ymin) / bev_res);
    int nz = floor((zmax - zmin) / bev_res) + 1;
    Eigen::MatrixXf bev_map = Eigen::MatrixXf::Zero(img_h * img_w, nz);
    std::vector<int> density_map(img_h * img_w);

    for (ssize_t n = 0; n < points.rows(); n++) {
        auto x = points(n, 0);
        if (x < xmin || x > xmax)
            continue;

        auto y = points(n, 1);
        if (y < ymin || y > ymax)
            continue;

        auto z = points(n, 2);
        if (z < zmin || z > zmax)
            continue;

        auto i = points(n, 3);
        // convert to ego-car coordinate
        int px = floor((x - xmin) / bev_res);
        int py = floor((ymax - y) / bev_res);
        int pz = floor((z - zmin) / bev_res);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);
        pz = std::min(pz, nz - 2);

        int idx = py * img_w + px;
        bev_map(idx, pz) = 1;
        bev_map(idx, nz - 1) += i;

        density_map[idx] += 1;
    }
    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            int idx = py * img_w + px;
            auto c = density_map[idx];
            if (c > 0) {
                bev_map(idx, nz - 1) /= c;
            }
        }
    }
    return bev_map;
}

PYBIND11_MODULE(bev, m)
{
    m.doc() = "Create BEV map from lidar points";

    m.def("rgb_map",
        &rgb_map,
        py::arg("points"),
        py::arg("lidar_range"),
        py::arg("bev_res"));

    m.def("height_map",
        &height_map,
        py::arg("points"),
        py::arg("lidar_range"),
        py::arg("bev_res"));

    m.def("rgb_traversability_map",
        &rgb_traversability_map,
        py::arg("points"),
        py::arg("lidar_range"),
        py::arg("sensor_height"),
        py::arg("bev_res"));
}
