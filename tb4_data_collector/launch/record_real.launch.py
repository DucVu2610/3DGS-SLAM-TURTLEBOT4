"""
record_real.launch.py

Thu du lieu RGB-D + quy dao tren TURTLEBOT 4 THAT voi camera RealSense D455.
(Ban cho mo phong Gazebo la record.launch.py - dung nham se khong ra du lieu.)

Khac biet chinh so voi ban sim:
  - Camera D455 (realsense2_camera) thay vi OAK-D mo phong
  - Dung /camera/camera/aligned_depth_to_color/image_raw: depth DA duoc
    driver align sang khung mau, cung 1280x720, khop pixel voi RGB
  - Pose lay tu /odom (co troi/drift theo thoi gian) thay vi
    /sim_ground_truth_pose (chinh xac tuyet doi, chi co trong sim)

Vi du:
    ros2 launch tb4_data_collector record_real.launch.py \
        robot_id:=robot1 output_dir:=/home/duc/tb4_dataset_real
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
    camera_frame = LaunchConfiguration('camera_frame')
    save_every_n = LaunchConfiguration('save_every_n')
    sync_slop = LaunchConfiguration('sync_slop')

    return LaunchDescription([
        DeclareLaunchArgument('namespace', default_value=''),
        DeclareLaunchArgument('robot_id', default_value='robot1'),
        DeclareLaunchArgument('output_dir', default_value='tb4_capture_real'),

        # --- Topic THAT cua RealSense D455 (da xac nhan bang ros2 topic list) ---
        DeclareLaunchArgument(
            'rgb_topic', default_value='/camera/camera/color/image_raw'),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/camera/aligned_depth_to_color/image_raw',
            description='Depth DA align sang khung mau - dung cai nay, KHONG dung '
                        '/camera/camera/depth/image_rect_raw (848x480, lech voi RGB)'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='/camera/camera/color/camera_info'),

        DeclareLaunchArgument(
            'odom_topic', default_value='/odom',
            description='Odometry tu Create3. Luu y: co troi (drift) theo thoi gian.'),
        DeclareLaunchArgument('base_frame', default_value='base_link'),
        DeclareLaunchArgument(
            'camera_frame', default_value='',
            description='De trong = lay tu header anh RGB (thuong la '
                        'camera_color_optical_frame)'),

        DeclareLaunchArgument('save_every_n', default_value='2',
                              description='Camera 30fps -> mac dinh luu 1/2 khung (~15fps) '
                                          'cho do nang. Dat 1 neu muon giu het.'),
        DeclareLaunchArgument('sync_slop', default_value='0.1'),

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
                    'camera_frame': camera_frame,
                    'robot_id': robot_id,
                    'output_dir': output_dir,
                    'save_every_n': save_every_n,
                    'sync_slop': sync_slop,
                    # Robot that dung gio he thong, KHONG phai /clock cua sim
                    'use_sim_time': False,
                }],
            ),
        ]),
    ])
