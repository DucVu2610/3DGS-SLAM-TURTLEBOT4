# tb4_data_collector

Package ROS 2 để thu thập dữ liệu **RGB-D + quỹ đạo (pose)** từ mô phỏng
TurtleBot 4 trên Gazebo, phục vụ dự án SLAM 3D dùng Gaussian Splatting
cho hệ đa robot.

Môi trường: **Ubuntu 22.04 + ROS 2 Humble + Ignition Gazebo Fortress**
(gói `turtlebot4_ignition_bringup`).

---

## 0. Troubleshooting môi trường (đọc trước nếu robot chưa spawn/di chuyển được)

Đây là các lỗi thực tế đã gặp và cách sửa, theo đúng thứ tự nên kiểm tra
nếu bạn thấy robot không spawn, không di chuyển, hoặc topic camera trống:

**a) Node ROS 2 không thấy nhau ngay trên cùng máy** (ví dụ `ros2 node
list` không ra gì, `talker`/`listener` demo không nhận được nhau) —
thường do card mạng Wi-Fi ở trạng thái `DORMANT` làm FastRTPS treo. Đổi
sang CycloneDDS:
```bash
sudo apt install ros-humble-rmw-cyclonedds-cpp
echo 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp' >> ~/.bashrc
echo 'export ROS_LOCALHOST_ONLY=1' >> ~/.bashrc
source ~/.bashrc
```

**b) Nhiều node Create3 (`motion_control`, `robot_state`, `ui_mgr`,...)
chết ngay khi khởi động** với lỗi `Failed to find a free participant
index for domain 0` — CycloneDDS giới hạn số participant mặc định quá
thấp so với ~40 tiến trình mà simulation TurtleBot4 khởi động cùng lúc.
Tăng giới hạn:
```bash
mkdir -p ~/.ros
cat > ~/.ros/cyclonedds.xml << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain>
    <Discovery>
      <MaxAutoParticipantIndex>200</MaxAutoParticipantIndex>
    </Discovery>
  </Domain>
</CycloneDDS>
XMLEOF
echo "export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml" >> ~/.bashrc
source ~/.bashrc
```

**c) `create-7`/`create-8` lặp mãi `Waiting messages on topic
[robot_description]`** — bình thường trong ~20-30 giây đầu (chờ DDS
discovery), chỉ là lỗi thật nếu vẫn lặp sau hơn 1 phút. Nếu vẫn lặp mãi,
99% là do (a) hoặc (b) ở trên chưa được áp dụng đúng trong terminal đang
chạy — kiểm tra `echo $RMW_IMPLEMENTATION` và `echo $CYCLONEDDS_URI`
ngay trong terminal đó trước khi launch.

**d) Robot spawn được nhưng `cmd_vel`/teleop không làm nó di chuyển**,
log hiện `Ignoring velocities commanded while an autonomous behavior is
running!` hoặc `Reached backup limit!` — do robot spawn đè/sát dock nên
cảm biến va chạm kích hoạt liên tục, hoặc do lệnh `/undock` chưa hoàn
tất. Cách né hẳn vấn đề này: **spawn robot cách dock ra một đoạn**
(không cần undock nữa):
```bash
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py \
    world:=maze x:=1.0 y:=0.0
```
Nếu vẫn cần undock (ví dụ robot phải bắt đầu từ dock), tắt tạm an toàn
lùi và gọi undock:
```bash
ros2 param set /motion_control safety_override full
ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}"
```

**e) Quy trình khởi động lại "sạch" khuyến nghị** mỗi khi đổi cấu hình:
```bash
pkill -9 -f motion_control; pkill -9 -f irobot_create; pkill -9 -f ign; pkill -9 -f gz; pkill -9 -f ros2
sleep 3
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py world:=maze x:=1.0 y:=0.0
```

## 1. Cài đặt package mô phỏng TurtleBot 4

```bash
sudo apt-get update && sudo apt-get install wget
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" \
    > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt-get update
sudo apt-get install ignition-fortress ros-humble-turtlebot4-simulator

sudo apt-get install \
    ros-humble-turtlebot4-description \
    ros-humble-turtlebot4-msgs \
    ros-humble-turtlebot4-navigation \
    ros-humble-turtlebot4-node \
    ros-humble-irobot-create-nodes

sudo apt-get install \
    ros-humble-cv-bridge ros-humble-message-filters \
    python3-opencv python3-numpy
```

## 2. Biến môi trường (`~/.bashrc`)

```bash
source /opt/ros/humble/setup.bash
source ~/tb4_ws/install/setup.bash

export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp     # xem mục 0.a
export ROS_LOCALHOST_ONLY=1                       # xem mục 0.a
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml   # xem mục 0.b
```

## 3. Bật mô phỏng

```bash
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py \
    world:=maze x:=1.0 y:=0.0
```
(`x:=1.0 y:=0.0` để robot spawn cách dock, tránh vấn đề ở mục 0.d)

Điều khiển thử:
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Đa robot

```bash
# Terminal 1 - robot 1 + world
ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py \
    namespace:=/robot1 world:=maze x:=1.0 y:=0.0

# Terminal 2 - robot 2, vị trí khác
ros2 launch turtlebot4_ignition_bringup turtlebot4_spawn.launch.py \
    namespace:=/robot2 x:=2.0 y:=0.0
```

## 4. Build package thu thập dữ liệu này

```bash
mkdir -p ~/tb4_ws/src
cp -r tb4_data_collector ~/tb4_ws/src/
cd ~/tb4_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select tb4_data_collector
source install/setup.bash
```

## 5. Tên topic thật (đã xác nhận trên bản build 06/2026)

```
/oakd/rgb/preview/image_raw      # RGB
/oakd/rgb/preview/depth          # Depth
/oakd/rgb/preview/camera_info    # Camera info
/sim_ground_truth_pose           # Pose thật (nav_msgs/Odometry), KHÔNG nhiễu/trôi
```
Tên topic có thể lệch nếu bạn dùng bản build khác - luôn kiểm tra lại bằng
`ros2 topic list | grep -i oakd` trước khi ghi dữ liệu thật.

Đây cũng chính là mặc định đã đặt sẵn trong `record.launch.py`.

## 6. Chạy node thu thập dữ liệu

```bash
ros2 launch tb4_data_collector record.launch.py \
    robot_id:=robot1 \
    output_dir:=/home/$USER/tb4_dataset
```

Với đa robot, thêm `namespace:=/robot1` (và `/robot2` ở lệnh thứ 2, chạy
song song terminal khác) để mỗi node tự động dùng đúng topic namespaced.

Cho robot di chuyển (teleop hoặc script riêng) để thu thập quỹ đạo.

Kết quả trong `~/tb4_dataset/robot1/`:

```
robot1/
├── rgb/000000.png, 000001.png, ...
├── depth/000000.png, ...          # PNG 16-bit, đơn vị milimet
├── transforms.json                 # định dạng nerfstudio, dùng cho Gaussian Splatting
└── poses_tum.txt                   # định dạng TUM, dùng đánh giá SLAM
```

Pose trong cả 2 file đều là **pose camera thật** trong world frame:
`pose_ground_truth_base_link` (từ `/sim_ground_truth_pose`) nhân với
offset tĩnh `base_link -> camera_optical_frame` (tra 1 lần qua TF lúc
khởi động node), sau đó chuyển sang quy ước trục của NeRF/Blender trong
`transforms.json`.

## 7. Lựa chọn thay thế / bổ sung

- **Chỉ cần dữ liệu thô**: `ros2 bag record -a` rồi xử lý offline.
- **Giảm dung lượng**: `save_every_n:=5` trong launch file.
- **Chuyển sang Jazzy sau này**: đổi `turtlebot4_ignition_bringup` →
  `turtlebot4_gz_bringup`, `turtlebot4_ignition.launch.py` →
  `turtlebot4_gz.launch.py`, `ignition-fortress` → `gz-harmonic`; node
  `tb4_data_collector` không cần sửa (miễn tên topic camera vẫn khớp).

---

# PHAN B — ROBOT THAT (TurtleBot 4 + RealSense D455)

Phan tren la cho mo phong Gazebo. Phan nay cho robot that.

## B1. Ket noi PC <-> robot (khac han sim!)

Cau hinh cho sim va cho robot that la HAI BO BIEN XUNG KHAC NHAU, khong
bat cung luc duoc. Tao 2 file rieng, source theo nhu cau:

```bash
# ~/tb4_sim_env.sh  (mo phong)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOCALHOST_ONLY=1
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT

# ~/tb4_real_env.sh  (robot that)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export ROS_DISCOVERY_SERVER="192.168.185.3:11811;"   # IP robot, NHO dau ';' cuoi
export ROS_SUPER_CLIENT=True
unset ROS_LOCALHOST_ONLY CYCLONEDDS_URI
```

Luu y quan trong:
- TB4 that dung **FastDDS + Discovery Server** chay tren chinh robot
  (`/etc/turtlebot4/setup.bash`). CycloneDDS dung cho sim se KHONG ket noi duoc.
- Dau `;` cuoi `ROS_DISCOVERY_SERVER` la bat buoc dung cu phap danh sach server.
- Ket noi bang **day Ethernet** on dinh hon Wi-Fi rat nhieu (ping 0.2ms
  so voi 24-735ms dao dong qua Wi-Fi). Dung day cho viec thu du lieu.
- Sau moi lan doi bien moi truong / doi IP, PHAI dop daemon roi thu lai:
  `pkill -9 -f _ros2_daemon; sleep 2; ros2 topic list`
  (`ros2 topic list` cache qua daemon, khong tu cap nhat khi doi config)

## B2. Chay camera RealSense D455 (tren ROBOT)

```bash
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description
sudo apt install ros-humble-diagnostic-updater   # neu bao thieu libdiagnostic_updater.so

# Chay trong tmux de node khong chet khi SSH dut
tmux new -s camera
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true pointcloud.enable:=true
# Ctrl+B roi D de thoat ma node van chay; quay lai: tmux attach -t camera
```

Topic thu duoc (da xac nhan tren D455 + realsense2_camera 4.58.3):
```
/camera/camera/color/image_raw                      # RGB 1280x720
/camera/camera/aligned_depth_to_color/image_raw     # Depth DA align, 1280x720, uint16 (mm)
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw                 # depth THO 848x480 - KHONG dung cho RGB-D
```

Luon dung ban **aligned_depth_to_color**: no khop pixel 1:1 voi anh mau.
Ban `depth/image_rect_raw` la depth goc 848x480, lech khung voi RGB.

## B3. Chay node thu du lieu (tren PC)

```bash
source ~/tb4_real_env.sh
source ~/tb4_ws/install/setup.bash
pkill -9 -f _ros2_daemon; sleep 2

ros2 launch tb4_data_collector record_real.launch.py \
    robot_id:=robot1 \
    output_dir:=/home/$USER/tb4_dataset_real
```

Cho robot di chuyen (teleop / Nav2) de thu quy dao. Ket qua giong phan sim:
`rgb/`, `depth/`, `transforms.json`, `poses_tum.txt`.

## B4. Hai han che can biet cua du lieu robot that

**1. Pose bi troi (drift).** `/odom` tren robot that tich luy sai so theo
thoi gian, khac han `/sim_ground_truth_pose` chinh xac tuyet doi trong sim.
Quy dao cang dai, sai so cang lon -> mo hinh Gaussian Splatting bi "nhoe".
Cach cai thien: chay SLAM song song (vd slam_toolbox) roi doi
`odom_topic` sang nguon pose da hieu chinh trong frame `map`.

**2. D455 chua co trong URDF cua TB4.** Khong co transform
`base_link -> camera_link`, nen node se tam dung pose THAN ROBOT thay cho
pose CAMERA (bo qua do lech lap dat). Neu camera lap lech nhieu so voi tam
robot, do tay roi khai bao static transform:

```bash
# x y z yaw pitch roll (met / radian), do tu tam base_link toi camera
ros2 run tf2_ros static_transform_publisher \
    0.1 0.0 0.2 0 0 0 base_link camera_link
```
Chay lenh nay TRUOC khi khoi dong node thu du lieu, node se tu tra duoc
transform va tinh dung pose camera.

## B5. Troubleshooting robot that (loi da gap that)

- **`ros2 topic list` chi ra 3 topic mac dinh** du ping thong: daemon cache
  sai. `pkill -9 -f _ros2_daemon` roi thu lai. Neu van thieu, kiem tra
  discovery server con song khong: tren robot `ss -tulnp | grep 11811`
  (khong ra gi = server chet, `sudo systemctl restart turtlebot4.service`).
- **Mat het topic dot ngot**: kiem tra day Ethernet (`ip addr show eth0`,
  `NO-CARRIER` = day tuot), va service (`systemctl status turtlebot4.service`).
- **apt bi treo `Waiting for cache lock`**: `unattended-upgrades` dang giu
  khoa. Xem PID bang `sudo fuser -v /var/lib/dpkg/lock-frontend`, roi
  `sudo kill -9 <PID>` (giet dung PID, `pkill` theo ten thuong khong an).
- **Robot mat Internet de cai goi**: `ip route` khong co dong `default` =
  chi co mang cuc bo. Bat Wi-Fi tam thoi:
  `sudo nmcli device wifi rescan && sudo nmcli device wifi connect "<SSID>" password "<pass>"`
  (Ethernet van dung song song cho ROS, khong xung dot.)

## B6. Bug mạng quan trọng nhất đã gặp: "Super Client" giả

Chỉ đặt `ROS_DISCOVERY_SERVER` (+ `ROS_SUPER_CLIENT=True`) là **CHƯA ĐỦ** để
PC thấy các topic **đã tồn tại trước** khi PC kết nối (`/battery_state`,
`/odom`, các topic camera). PC chỉ thấy được entity **mới tạo sau** khi nó
kết nối (ví dụ 1 talker mới chạy) — dễ nhầm tưởng "mạng đã thông" trong khi
PC thực ra vẫn ở chế độ Client thường, chưa phải Super Client thật sự.
Nguyên nhân: `ROS_SUPER_CLIENT` chỉ có tác dụng khi discovery protocol ban
đầu là SIMPLE, nhưng `ROS_DISCOVERY_SERVER` đã tự chuyển nó sang CLIENT
trước đó mất.

**Fix đúng cách — dùng file XML định nghĩa SUPER_CLIENT rõ ràng, KHÔNG dùng
chung với `ROS_DISCOVERY_SERVER`:**

```bash
cat > ~/super_client_configuration_file.xml << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8" ?>
<dds>
    <profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
        <participant profile_name="super_client_profile" is_default_profile="true">
            <rtps>
                <builtin>
                    <discovery_config>
                        <discoveryProtocol>SUPER_CLIENT</discoveryProtocol>
                        <discoveryServersList>
                            <RemoteServer prefix="44.53.00.5f.45.50.52.4f.53.49.4d.41">
                                <metatrafficUnicastLocatorList>
                                    <locator>
                                        <udpv4>
                                            <address>10.42.0.1</address>
                                            <port>11811</port>
                                        </udpv4>
                                    </locator>
                                </metatrafficUnicastLocatorList>
                            </RemoteServer>
                        </discoveryServersList>
                    </discovery_config>
                </builtin>
            </rtps>
        </participant>
    </profiles>
</dds>
XMLEOF
```
(prefix `44.53.00...` ứng với server ID 0, khớp cách robot khởi động
`fast-discovery-server -i 0`; đổi `10.42.0.1` nếu IP robot khác)

```bash
# tb4_real_env.sh - KHÔNG đặt ROS_DISCOVERY_SERVER khi đã dùng XML này
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/super_client_configuration_file.xml
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset ROS_LOCALHOST_ONLY CYCLONEDDS_URI ROS_SUPER_CLIENT ROS_DISCOVERY_SERVER
```

**Cách chẩn đoán phân biệt "Client thường" vs "thật sự thấy hết graph":**
chạy talker MỚI trên PC, xem robot có thấy không (luôn thấy, không nói lên
gì) → rồi thử ngay `ros2 topic echo /battery_state --once` trên PC (topic
CÓ SẴN TỪ TRƯỚC) — nếu cái sau không ra gì dù cái trước thành công, đúng là
thiếu Super Client thật sự.

## B7. Bug đã biết khác cần nhớ

- **`create3_republisher` trong `robot.launch.py` bị lỗi thiếu dấu `/`**
  (`'_do_not_use'` thay vì `'/_do_not_use'`), gây `InvalidTopicNameError` và
  crash mỗi khi `turtlebot4.service` khởi động lại → `/odom` biến mất. Fix
  tạm (chạy tay sau mỗi lần service restart/reboot):
  ```bash
  ros2 run create3_republisher create3_republisher --ros-args \
      -r __ns:=/ \
      -p robot_namespace:=/_do_not_use \
      --params-file /opt/ros/humble/share/create3_republisher/bringup/params.yaml
  ```
  Fix vĩnh viễn (rủi ro: mất khi `apt upgrade`):
  ```bash
  sudo sed -i "s/'_do_not_use'/'\/_do_not_use'/" \
      /opt/ros/humble/share/turtlebot4_bringup/launch/robot.launch.py
  sudo systemctl restart turtlebot4.service
  ```
- **D455 rớt kết nối USB định kỳ** (`UVCIOC_CTRL_QUERY ... Connection timed
  out`) — rút/cắm lại cáp USB-C, không phải lỗi phần mềm.
- **Nhiều tiến trình `realsense2_camera_node` chạy song song** nếu Ctrl+C
  launch wrapper mà tiến trình con không chết theo — luôn
  `ps aux | grep realsense` trước khi chạy lại, `pkill -9 -f
  realsense2_camera_node` nếu thấy nhiều hơn 1.
- **turtlebot4-setup đổi Wi-Fi AP mode xong vẫn không phát** nếu còn 1
  connection Wi-Fi client cũ tự tạo bằng `nmcli` với `autoconnect` bật —
  `nmcli connection delete <ten_cu>` rồi `nmcli connection up
  netplan-wlan0-<SSID_AP>`.
