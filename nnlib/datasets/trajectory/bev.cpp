#include <vector>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> create_bev(
        py::array_t<double> points, /* Nx4 ndarray */
        py::array_t<double> lidar_range, /* (6,) ndarray, [xmin, xmax, ymin, ymax, zmin, zmax] */
        int img_h, int img_w, /* bev image size */
        double bev_res /* m per pixel*/) {
    if (points.ndim() != 2)
        throw std::runtime_error("ndim of points must be 2");
    
    if (lidar_range.ndim() != 1)
        throw std::runtime_error("ndim of lidar_range must be 1");

    if (lidar_range.size() != 6)
        throw std::runtime_error("size of lidar_range must be 6");

    auto l = lidar_range.unchecked<1>(); // ndim = 1
    auto xmin = l[0];
    auto xmax = l[1];
    auto ymin = l[2];
    auto ymax = l[3];
    auto zmin = l[4];
    auto zmax = l[5];
    
    int nz = int((zmax - zmin) / bev_res) + 1;
    auto bev_map = py::array_t<double>(img_h * img_w * nz);
    bev_map.resize({img_h, img_w, nz});

    py::buffer_info buf = bev_map.request();
    double *ptr = static_cast<double *>(buf.ptr);
    memset(ptr, 0.0, sizeof(double) * buf.size);

    auto r = bev_map.mutable_unchecked<3>();

    std::vector<int> density_map(img_h * img_w);

    auto p = points.unchecked<2>(); // points must have ndim = 3
    
    if (p.shape(1) != 4)
        throw std::runtime_error("axis 1 of points must have size 4");

    for (ssize_t n = 0; n < p.shape(0); n++) {
        auto x = p(n, 0);
        if (x < xmin || x > xmax)
            continue;
        
        auto y = p(n, 1);
        if (y < ymin || y > ymax)
            continue;

        auto z = p(n, 2);
        if (z < zmin || z > zmax)
            continue;
        
        auto i = p(n, 3);
        // convert to ego-car coordinate
        int px = int((x - xmin) / bev_res);
        int py = int((ymax - y) / bev_res);
        int pz = int((z - zmin) / bev_res);
        px = std::min(px, img_w - 1);
        py = std::min(py, img_h - 1);
        pz = std::min(pz, nz - 2);
        // std::cout << "px: " << px << ", py: " << py << ", pz: " << pz << ", i: " << i << std::endl;

        r(py, px, pz) = 1;
        r(py, px, nz - 1) += i;

        int idx = py * img_w + px;
        density_map[idx] += 1;
    }
    for (ssize_t py = 0; py < img_h; py++) {
        for (ssize_t px = 0; px < img_w; px++) {
            auto c = density_map[py * img_w + px];
            if (c > 0) {
                r(py, px, nz - 1) /= c;
            }
        }
    }
    return bev_map;
}

PYBIND11_MODULE(bev, m) {
    m.doc() = "Create BEV map from lidar points"; // optional module docstring

    m.def("create_bev", &create_bev, "A function which converts point cloud to BEV map");
}
