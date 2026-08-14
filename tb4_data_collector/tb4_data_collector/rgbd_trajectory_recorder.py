#!/usr/bin/env python3
"""
rgbd_trajectory_recorder.py

Đồng bộ RGB + Depth + CameraInfo + pose ground-truth (/sim_ground_truth_pose,
nav_msgs/Odometry) từ mô phỏng TurtleBot4/Ignition Gazebo, rồi ghi ra đĩa:

    <output_dir>/<robot_id>/
        rgb/000000.png, 000001.png, ...
        depth/000000.png, ...      (PNG 16-bit, đơn vị = mm)
        transforms.json             (định dạng kiểu NeRF/3DGS - nerfstudio)
        poses_tum.txt                (định dạng TUM: t tx ty tz qx qy qz qw)

Vì sao dùng /sim_ground_truth_pose thay vì tra TF theo thời gian: pose này
là pose thật của robot lấy trực tiếp từ Gazebo (không nhiễu, không trôi),
và không phụ thuộc vào việc TF buffer có đủ lịch sử tại đúng timestamp hay
không - tránh toàn bộ lớp lỗi "extrapolation into the past" hay gặp khi
đồng bộ camera tốc độ thấp với TF cập nhật nhanh.

Pose ground-truth là pose của base_link trong world. Để ra đúng pose CAMERA
(cần cho Gaussian Splatting), node tự tra thêm 1 lần transform TĨNH
base_link -> camera_optical_frame qua TF (transform này không đổi theo thời
gian nên chỉ cần lấy 1 lần lúc khởi động, không có rủi ro timing).
"""

import os
import json

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import message_filters
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import tf2_ros
from tf2_ros import TransformException


# Ma trận chuyển hệ quy chiếu: ROS optical frame (X phải, Y xuống, Z hướng
# tới trước ống kính) -> quy ước camera kiểu NeRF/Blender/nerfstudio
# (X phải, Y lên, Z hướng ra sau ống kính).
ROS_OPTICAL_TO_NERF = np.array([
    [1.0,  0.0,  0.0, 0.0],
    [0.0, -1.0,  0.0, 0.0],
    [0.0,  0.0, -1.0, 0.0],
    [0.0,  0.0,  0.0, 1.0],
], dtype=np.float64)


def quat_to_matrix(x, y, z, w):
    """Quaternion (x, y, z, w) -> ma trận xoay 3x3."""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    X, Y, Z, W = x * s, y * s, z * s, w * s
    xx, xy, xz = x * X, x * Y, x * Z
    yy, yz, zz = y * Y, y * Z, z * Z
    wx, wy, wz = w * X, w * Y, w * Z
    return np.array([
        [1.0 - (yy + zz),       xy - wz,             xz + wy],
        [      xy + wz,   1.0 - (xx + zz),           yz - wx],
        [      xz - wy,         yz + wx,       1.0 - (xx + yy)],
    ])


def pose_to_matrix(position, orientation):
    """geometry_msgs Point + Quaternion -> ma trận 4x4."""
    R = quat_to_matrix(orientation.x, orientation.y, orientation.z, orientation.w)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [position.x, position.y, position.z]
    return T


class RgbdTrajectoryRecorder(Node):

    def __init__(self):
        super().__init__('rgbd_trajectory_recorder')

        # ---------------- Tham số ----------------
        self.declare_parameter('rgb_topic', 'oakd/rgb/preview/image_raw')
        self.declare_parameter('depth_topic', 'oakd/rgb/preview/depth')
        self.declare_parameter('camera_info_topic', 'oakd/rgb/preview/camera_info')
        self.declare_parameter('odom_topic', 'sim_ground_truth_pose')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', '')   # rỗng = lấy từ header ảnh RGB
        self.declare_parameter('output_dir', 'tb4_capture')
        self.declare_parameter('robot_id', 'robot1')
        self.declare_parameter('sync_slop', 0.1)     # giây, camera preview có thể chậm hơn odom
        self.declare_parameter('depth_scale', 1000.0)
        self.declare_parameter('save_every_n', 1)

        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        info_topic = self.get_parameter('camera_info_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame_override = self.get_parameter('camera_frame').value
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.save_every_n = max(1, int(self.get_parameter('save_every_n').value))

        robot_id = self.get_parameter('robot_id').value
        out_root = self.get_parameter('output_dir').value
        self.out_dir = os.path.join(out_root, robot_id)
        self.rgb_dir = os.path.join(self.out_dir, 'rgb')
        self.depth_dir = os.path.join(self.out_dir, 'depth')
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.T_base_camera = None          # transform tĩnh, tra 1 lần
        self._tf_warned = False

        self._frame_idx = 0
        self._msg_count = 0
        self.intrinsics = None
        self._nerf_frames = []
        self._tum_rows = []

        # ---------------- Subscriber đồng bộ ----------------
        # Nếu odom_topic để trống -> CHẾ ĐỘ KHÔNG POSE: chỉ đồng bộ 3 luồng
        # camera, ghi ảnh + timestamp + intrinsics. Quỹ đạo sẽ được tái tạo
        # offline sau bằng COLMAP/RTAB-Map từ chính chuỗi ảnh. Hữu ích khi
        # nguồn odometry chưa sẵn sàng, và với Gaussian Splatting thì pose
        # từ visual SfM thường còn chính xác hơn odometry bánh xe.
        self.no_pose_mode = not bool(odom_topic)

        self.sub_rgb = message_filters.Subscriber(
            self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = message_filters.Subscriber(
            self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sub_info = message_filters.Subscriber(
            self, CameraInfo, info_topic, qos_profile=qos_profile_sensor_data)

        subs = [self.sub_rgb, self.sub_depth, self.sub_info]
        if not self.no_pose_mode:
            self.sub_odom = message_filters.Subscriber(
                self, Odometry, odom_topic, qos_profile=qos_profile_sensor_data)
            subs.append(self.sub_odom)

        slop = float(self.get_parameter('sync_slop').value)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=30, slop=slop)
        if self.no_pose_mode:
            self.ts.registerCallback(self.callback_no_pose)
        else:
            self.ts.registerCallback(self.callback)

        self.create_timer(5.0, self._flush)

        self.get_logger().info(
            "Bắt đầu ghi dữ liệu cho '%s'\n"
            "  RGB   : %s\n  Depth : %s\n  Info  : %s\n  Odom  : %s\n"
            "  Lưu vào: %s"
            % (robot_id, rgb_topic, depth_topic, info_topic, odom_topic, self.out_dir)
        )

    # ------------------------------------------------------------------
    def _get_static_base_to_camera(self, camera_frame):
        """Tra transform TĨNH base_link -> camera_optical_frame, chỉ cần 1 lần."""
        if self.T_base_camera is not None:
            return self.T_base_camera
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, camera_frame, Time())  # Time() = "mới nhất có sẵn"
        except TransformException as ex:
            if not self._tf_warned:
                self.get_logger().warn(
                    f"Chưa tra được TF tĩnh '{self.base_frame}' -> '{camera_frame}' "
                    f"({ex}). Tạm dùng pose thân robot (bỏ qua offset camera) "
                    f"cho tới khi tra được.")
                self._tf_warned = True
            return None
        self.T_base_camera = pose_to_matrix(tf.transform.translation, tf.transform.rotation)
        self.get_logger().info(f"Đã lấy được offset camera so với '{self.base_frame}'.")
        return self.T_base_camera

    def _write_images(self, rgb_msg, depth_msg):
        """Chuyển ảnh ROS -> file PNG trên đĩa. Trả về (tên_file, ảnh_rgb)."""
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # RealSense xuất depth sẵn dạng uint16 (mm) -> giữ nguyên.
        # Gazebo/một số driver xuất float32 (mét) -> nhân depth_scale.
        if depth_raw.dtype != np.uint16:
            depth_m = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
            depth_mm = np.clip(depth_m * self.depth_scale, 0, 65535).astype(np.uint16)
        else:
            depth_mm = depth_raw

        fname = f"{self._frame_idx:06d}.png"
        cv2.imwrite(os.path.join(self.rgb_dir, fname), rgb)
        cv2.imwrite(os.path.join(self.depth_dir, fname), depth_mm)
        return fname, rgb

    def callback_no_pose(self, rgb_msg, depth_msg, info_msg):
        """Chế độ không pose: chỉ ghi ảnh + timestamp + intrinsics."""
        self._msg_count += 1
        if (self._msg_count - 1) % self.save_every_n != 0:
            return

        fname, rgb = self._write_images(rgb_msg, depth_msg)

        if self.intrinsics is None:
            self._save_intrinsics(info_msg, rgb.shape[1], rgb.shape[0])

        stamp = rgb_msg.header.stamp
        self._nerf_frames.append({
            "file_path": f"rgb/{fname}",
            "depth_file_path": f"depth/{fname}",
            "timestamp": stamp.sec + stamp.nanosec * 1e-9,
            # Không có transform_matrix: pose sẽ tái tạo offline bằng
            # COLMAP / RTAB-Map từ chính chuỗi ảnh này.
        })

        self._frame_idx += 1
        if self._frame_idx % 50 == 0:
            self.get_logger().info(f"Đã lưu {self._frame_idx} khung hình (không pose)...")

    def callback(self, rgb_msg, depth_msg, info_msg, odom_msg):
        self._msg_count += 1
        if (self._msg_count - 1) % self.save_every_n != 0:
            return

        camera_frame = self.camera_frame_override or rgb_msg.header.frame_id
        T_world_base = pose_to_matrix(odom_msg.pose.pose.position, odom_msg.pose.pose.orientation)

        T_base_camera = self._get_static_base_to_camera(camera_frame)
        T_world_camera = T_world_base if T_base_camera is None else T_world_base @ T_base_camera

        fname, rgb = self._write_images(rgb_msg, depth_msg)

        stamp = rgb_msg.header.stamp
        t_sec = stamp.sec + stamp.nanosec * 1e-9
        p = odom_msg.pose.pose.position
        q = odom_msg.pose.pose.orientation
        self._tum_rows.append((t_sec, p.x, p.y, p.z, q.x, q.y, q.z, q.w))

        if self.intrinsics is None:
            self._save_intrinsics(info_msg, rgb.shape[1], rgb.shape[0])

        T_nerf = T_world_camera @ ROS_OPTICAL_TO_NERF
        self._nerf_frames.append({
            "file_path": f"rgb/{fname}",
            "depth_file_path": f"depth/{fname}",
            "timestamp": t_sec,
            "transform_matrix": T_nerf.tolist(),
        })

        self._frame_idx += 1
        if self._frame_idx % 50 == 0:
            self.get_logger().info(f"Đã lưu {self._frame_idx} khung hình...")

    # ------------------------------------------------------------------
    def _save_intrinsics(self, info_msg, width, height):
        K = info_msg.k
        self.intrinsics = {
            "fl_x": float(K[0]), "fl_y": float(K[4]),
            "cx": float(K[2]), "cy": float(K[5]),
            "w": int(width), "h": int(height),
            "depth_scale": self.depth_scale,
            "camera_model": "OPENCV",
        }

    def _flush(self):
        if self.intrinsics is not None and self._nerf_frames:
            data = dict(self.intrinsics)
            data["frames"] = self._nerf_frames
            with open(os.path.join(self.out_dir, 'transforms.json'), 'w') as f:
                json.dump(data, f, indent=2)

        if self._tum_rows:
            with open(os.path.join(self.out_dir, 'poses_tum.txt'), 'w') as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                for row in self._tum_rows:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

    def destroy_node(self):
        self._flush()
        self.get_logger().info(f"Đã ghi tổng cộng {self._frame_idx} khung hình vào {self.out_dir}.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RgbdTrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
