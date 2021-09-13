import os
import sys
from setuptools import Extension
from setuptools import setup, find_packages

__version__ = '0.0.0'
        
py_dir = '/usr/local/lib/python{}/dist-packages/'.format(sys.version[:3])

# lidar BEV cpp extension
lidar_bev_ext_module = Extension(
    name = 'lidardet.ops.lidar_bev.bev',
    sources = ['lidardet/ops/lidar_bev/bev.cpp'], 
    include_dirs = ['/usr/include/eigen3', py_dir + 'pybind11/include'],
    language='c++'
)

if __name__ == '__main__':

    setup(
        name='lidardet',
        version="0.0.0",
        description='LidarDet is a general codebase for learning based perception task',
        install_requires=[
            'numpy',
            'torch>=1.1',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='jiaolong',
        author_email='jiaolongxu@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output', 'cache', 'ros', 'docker', 'config']),
        ext_modules=[lidar_bev_ext_module]
        )
