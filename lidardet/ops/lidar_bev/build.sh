c++ -O3 -Wall -shared -std=c++11 -fPIC -I /usr/include/eigen3 `python -m pybind11 --includes` bev.cpp -o bev`python3.6-config --extension-suffix`
