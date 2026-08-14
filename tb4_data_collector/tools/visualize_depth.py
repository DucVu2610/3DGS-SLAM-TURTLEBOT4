#!/usr/bin/env python3
"""
visualize_depth.py

Chuyển đổi ảnh depth 16-bit (đơn vị mm, do rgbd_trajectory_recorder lưu ra)
thành ảnh xem được (8-bit, có màu) để kiểm tra chất lượng dữ liệu bằng mắt.

QUAN TRỌNG: đây CHỈ là công cụ xem/kiểm tra (QC) - không sửa đổi hay ghi đè
lên file depth gốc. Ảnh depth gốc trong dataset của bạn giữ nguyên độ chính
xác 16-bit đầy đủ, không bị mất dữ liệu.

Vì sao ảnh depth mở bằng trình xem ảnh thường lại nhìn gần như đen: giá trị
lưu là milimet (uint16, 0-65535), nhưng khoảng cách trong nhà thường chỉ
500-6000mm - chỉ chiếm ~1-9% thang sáng tối đa, nên mắt người thấy toàn màu
đen dù dữ liệu hoàn toàn đúng.

Cách dùng:
    # Xem thử 1 ảnh, giới hạn hiển thị ở 6000mm (mặc định)
    python3 visualize_depth.py rgb_depth_pair --depth path/to/000175.png

    # Xuất cả thư mục depth/ ra thư mục preview/ (ảnh màu, dễ xem)
    python3 visualize_depth.py batch --depth-dir tb4_dataset_real/robot1/depth \
        --out-dir tb4_dataset_real/robot1/depth_preview --max-depth-mm 6000

    # In thống kê nhanh (min/max/mean/% pixel có dữ liệu) không cần xuất ảnh
    python3 visualize_depth.py stats --depth-dir tb4_dataset_real/robot1/depth
"""

import argparse
import os
import sys

import numpy as np
import cv2


def load_depth(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được: {path}")
    if img.dtype != np.uint16:
        raise ValueError(f"{path} không phải PNG 16-bit (dtype={img.dtype})")
    return img


def depth_to_viewable(depth_mm, max_depth_mm=6000, colormap=cv2.COLORMAP_TURBO):
    """
    Chuẩn hóa depth (uint16, mm) về 8-bit để xem được.
    Giá trị 0 (không đo được) luôn hiển thị màu đen, tách biệt với "gần".
    """
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, 0, max_depth_mm).astype(np.float32)
    normalized = (clipped / max_depth_mm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, colormap)
    colored[~valid] = (0, 0, 0)  # pixel không có dữ liệu -> đen, không lẫn với "gần"
    return colored


def cmd_rgb_depth_pair(args):
    depth = load_depth(args.depth)
    viewable = depth_to_viewable(depth, args.max_depth_mm)
    out_path = args.out or (os.path.splitext(args.depth)[0] + "_preview.png")
    cv2.imwrite(out_path, viewable)
    print(f"Đã lưu ảnh xem được: {out_path}")
    print(f"  min={depth.min()}mm max={depth.max()}mm mean={depth[depth>0].mean():.1f}mm "
          f"(chỉ tính pixel có dữ liệu)")
    print(f"  % pixel có dữ liệu: {100*np.count_nonzero(depth)/depth.size:.1f}%")


def cmd_batch(args):
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.depth_dir) if f.lower().endswith('.png'))
    if not files:
        print(f"Không tìm thấy PNG nào trong {args.depth_dir}")
        sys.exit(1)
    for i, fname in enumerate(files):
        depth = load_depth(os.path.join(args.depth_dir, fname))
        viewable = depth_to_viewable(depth, args.max_depth_mm)
        cv2.imwrite(os.path.join(args.out_dir, fname), viewable)
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            print(f"  {i+1}/{len(files)} ảnh đã xử lý...")
    print(f"Xong. {len(files)} ảnh xem được đã lưu vào {args.out_dir}")
    print(f"(max_depth_mm={args.max_depth_mm} -> pixel càng xa màu càng khác, "
          f"pixel không có dữ liệu = đen tuyền)")


def cmd_stats(args):
    files = sorted(f for f in os.listdir(args.depth_dir) if f.lower().endswith('.png'))
    if not files:
        print(f"Không tìm thấy PNG nào trong {args.depth_dir}")
        sys.exit(1)
    print(f"{'file':<16} {'min(mm)':>8} {'max(mm)':>8} {'mean_valid(mm)':>15} {'%valid':>8}")
    all_valid_pct = []
    for fname in files:
        depth = load_depth(os.path.join(args.depth_dir, fname))
        valid = depth[depth > 0]
        pct = 100 * valid.size / depth.size
        all_valid_pct.append(pct)
        mean_valid = valid.mean() if valid.size else 0.0
        print(f"{fname:<16} {depth.min():>8} {depth.max():>8} {mean_valid:>15.1f} {pct:>7.1f}%")
    print(f"\nTrung bình % pixel có dữ liệu trên cả tập: {np.mean(all_valid_pct):.1f}%")
    print("Gợi ý: nếu %valid liên tục dưới ~50-60%, kiểm tra lại bề mặt quét "
          "(vật phản chiếu/kính/quá xa/quá gần D455 dễ mất depth).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p1 = sub.add_parser('rgb_depth_pair', help='Xem thử 1 ảnh depth')
    p1.add_argument('--depth', required=True, help='Đường dẫn ảnh depth PNG 16-bit')
    p1.add_argument('--out', default=None, help='Đường dẫn lưu ảnh xem được (mặc định: <tên>_preview.png)')
    p1.add_argument('--max-depth-mm', type=float, default=6000, help='Khoảng cách tối đa hiển thị, mm (mặc định 6000)')
    p1.set_defaults(func=cmd_rgb_depth_pair)

    p2 = sub.add_parser('batch', help='Xuất cả thư mục depth/ ra ảnh xem được')
    p2.add_argument('--depth-dir', required=True)
    p2.add_argument('--out-dir', required=True)
    p2.add_argument('--max-depth-mm', type=float, default=6000)
    p2.set_defaults(func=cmd_batch)

    p3 = sub.add_parser('stats', help='In thống kê min/max/%%valid cho cả thư mục depth/')
    p3.add_argument('--depth-dir', required=True)
    p3.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
