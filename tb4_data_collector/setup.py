import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'tb4_data_collector'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='you@example.com',
    description=(
        'Ghi du lieu RGB-D dong bo + quy dao tu mo phong TurtleBot4/Gazebo, '
        'phuc vu SLAM va Gaussian Splatting.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_trajectory_recorder = tb4_data_collector.rgbd_trajectory_recorder:main',
        ],
    },
)
