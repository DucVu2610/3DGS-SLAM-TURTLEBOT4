"""
record.launch.py

Chạy node rgbd_trajectory_recorder bên trong namespace của MỘT robot.
Với hệ đa robot, gọi launch file này nhiều lần (mỗi lần một namespace/
robot_id/output_dir khác nhau).

Ví dụ:
    ros2 launch tb4_data_collector record.launch.py \
        robot_id:=robot1 output_dir:=/home/duc/tb4_dataset
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace


def generate_launch_description():
    namespace = LaunchConfiguration('namespace')
    robot_id = LaunchConfiguration('robot_id')
    output_dir = LaunchConfiguration('output_dir')
    rgb_topic = LaunchConfiguration('rgb_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    odom_topic = LaunchConfiguration('odom_topic')
    base_frame = LaunchConfiguration('base_frame')
    save_every_n = LaunchConfiguration('save_every_n')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value='',
                               description='Namespace của robot, vd: /robot1'),
        DeclareLaunchArgument('robot_id', default_value='robot1',
                               description='Tên thư mục con lưu dữ liệu của robot này'),
        DeclareLaunchArgument('output_dir', default_value='tb4_capture',
                               description='Thư mục gốc lưu dữ liệu'),
        DeclareLaunchArgument('rgb_topic', default_value='oakd/rgb/preview/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='oakd/rgb/preview/depth'),
        DeclareLaunchArgument('camera_info_topic', default_value='oakd/rgb/preview/camera_info'),
        DeclareLaunchArgument('odom_topic', default_value='sim_ground_truth_pose',
                               description='Pose ground-truth từ Gazebo (nav_msgs/Odometry)'),
        DeclareLaunchArgument('base_frame', default_value='base_link',
                               description='Frame gốc thân robot dùng để tra offset camera qua TF'),
        DeclareLaunchArgument('save_every_n', default_value='1',
                               description='Chỉ lưu 1/N khung hình để giảm dung lượng'),

        GroupAction([
            PushRosNamespace(namespace),
            Node(
                package='tb4_data_collector',
                executable='rgbd_trajectory_recorder',
                name='rgbd_trajectory_recorder',
                output='screen',
                parameters=[{
                    'rgb_topic': rgb_topic,
                    'depth_topic': depth_topic,
                    'camera_info_topic': camera_info_topic,
                    'odom_topic': odom_topic,
                    'base_frame': base_frame,
                    'robot_id': robot_id,
                    'output_dir': output_dir,
                    'save_every_n': save_every_n,
                    'use_sim_time': True,
                }],
            ),
        ]),
    ])
