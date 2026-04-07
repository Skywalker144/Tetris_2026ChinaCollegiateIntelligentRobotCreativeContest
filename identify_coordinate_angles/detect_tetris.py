import cv2
import numpy as np
import math
import sys

# =============================================================================
# 物理参数 (Physical Parameters)
# =============================================================================
BLOCK_SIZE_MM = 18      # 单个基础块边长 (mm)
GAP_MM = 2              # 基础块之间间距 (mm)
BLOCK_PITCH_MM = BLOCK_SIZE_MM + GAP_MM  # 20mm, 块中心到相邻块中心的距离

# =============================================================================
# HSV 颜色范围
# O(Orange) 和 T(Brown) 的 H 值几乎相同 (~12), 需要用 V(明度) 区分:
#   O: V > 200 (明亮橙色)
#   T: V < 195 (暗棕色)
# =============================================================================
COLOR_RANGES = {
    'J': [([125, 80, 80], [150, 255, 255])],                                    # 紫色
    'L': [([22, 100, 100], [38, 255, 255])],                                     # 黄色
    'Z': [([90, 100, 80], [115, 255, 255])],                                     # 蓝色
    'N': [([60, 100, 80], [90, 255, 255])],                                      # 绿色
    'O': [([5, 150, 200], [20, 255, 255])],                                      # 橙色 (高明度)
    'T': [([5, 100, 50], [20, 255, 195])],                                       # 棕色 (低明度)
    'I': [([170, 80, 80], [180, 255, 255]), ([0, 80, 80], [5, 255, 255])],       # 红色
}


def classify_by_geometry(w, h, solidity):
    """
    根据 minAreaRect 的长宽比和 solidity 粗分类。
    w > h 已保证。
    """
    aspect = w / h if h > 0 else 0
    if aspect > 3.2:
        return 'I'          # 长条型 4:1
    elif aspect < 1.15 and solidity > 0.95:
        return 'O'          # 正方形 1:1
    else:
        return '4block'     # T, L, J, Z, N: aspect ~1.5, solidity ~0.8


def find_target_center_by_grid(cnt, shape_type, mask_shape):
    """
    通用的目标中心点定位函数 (栅格化分析法)。

    核心思路:
    1. 利用 minAreaRect 将方块旋转到水平(0度)
    2. 裁剪出方块区域
    3. 将裁剪区域划分为 rows x cols 的栅格 (如 2x3)
    4. 分析每个格子的占用率, 得到二值化网格
    5. 根据网格布局找到目标块:
       - L/J: 有 2 个垂直邻居的块 (转角块)
       - T: 有 3 个邻居的块 (中间块)
       - Z/N/I/O: 所有被占用格子的几何平均中心
    6. 将目标块的坐标逆旋转回原图

    参数:
        cnt: 轮廓
        shape_type: 形状类型 ('L', 'J', 'T', 'Z', 'N', 'I', 'O')
        mask_shape: 原图 mask 的 shape (height, width)
    返回:
        ((x, y), binary_grid): 原图中心坐标 和 二值化栅格 (用于角度计算)
    """
    rect = cv2.minAreaRect(cnt)
    center, (rw, rh), angle = rect

    # 确保 rw >= rh
    if rw < rh:
        rw, rh = rh, rw
        angle += 90
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180

    # --- 步骤 1: 创建填充 mask 并旋转到 0 度 ---
    single_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(single_mask, [cnt], -1, 255, -1)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    h_img, w_img = mask_shape
    rotated = cv2.warpAffine(single_mask, M, (w_img, h_img))

    # --- 步骤 2: 裁剪 ---
    margin = 5
    cx, cy = int(center[0]), int(center[1])
    half_w, half_h = int(rw / 2) + margin, int(rh / 2) + margin
    x1 = max(0, cx - half_w)
    y1 = max(0, cy - half_h)
    x2 = min(w_img, cx + half_w)
    y2 = min(h_img, cy + half_h)
    crop = rotated[y1:y2, x1:x2]

    if crop.size == 0:
        Mo = cv2.moments(cnt)
        if Mo['m00'] > 0:
            return ((Mo['m10'] / Mo['m00'], Mo['m01'] / Mo['m00']), None)
        return ((center[0], center[1]), None)

    crop_h, crop_w = crop.shape

    # --- 步骤 3: 确定栅格尺寸 ---
    if shape_type == 'I':
        grid_rows, grid_cols = 1, 4
    elif shape_type == 'O':
        grid_rows, grid_cols = 2, 2
    else:
        grid_rows, grid_cols = 2, 3

    # --- 步骤 4: 计算每个格子的占用率 ---
    grid = np.zeros((grid_rows, grid_cols))
    for r in range(grid_rows):
        for c in range(grid_cols):
            gy1 = int(r * crop_h / grid_rows)
            gy2 = int((r + 1) * crop_h / grid_rows)
            gx1 = int(c * crop_w / grid_cols)
            gx2 = int((c + 1) * crop_w / grid_cols)
            cell = crop[gy1:gy2, gx1:gx2]
            if cell.size > 0:
                grid[r, c] = np.sum(cell > 0) / cell.size

    binary = (grid > 0.5).astype(int)

    # --- 步骤 5: 根据形状类型找到目标格子 ---
    occupied = np.argwhere(binary == 1)
    target_r, target_c = -1, -1

    if shape_type in ['L', 'J']:
        for (r, c) in occupied:
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_rows and 0 <= nc < grid_cols and binary[nr, nc] == 1:
                    neighbors.append((dr, dc))
            if len(neighbors) == 2:
                (d1r, d1c), (d2r, d2c) = neighbors
                if d1r * d2r + d1c * d2c == 0:
                    target_r, target_c = r, c
                    break

    elif shape_type == 'T':
        for (r, c) in occupied:
            neighbor_count = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_rows and 0 <= nc < grid_cols and binary[nr, nc] == 1:
                    neighbor_count += 1
            if neighbor_count == 3:
                target_r, target_c = r, c
                break

    # --- 步骤 6: 计算目标中心并逆旋转回原图 ---
    if target_r >= 0 and target_c >= 0:
        cell_w = crop_w / grid_cols
        cell_h = crop_h / grid_rows
        target_cx = (target_c + 0.5) * cell_w
        target_cy = (target_r + 0.5) * cell_h
    else:
        if len(occupied) > 0:
            cell_w = crop_w / grid_cols
            cell_h = crop_h / grid_rows
            avg_c = np.mean(occupied[:, 1])
            avg_r = np.mean(occupied[:, 0])
            target_cx = (avg_c + 0.5) * cell_w
            target_cy = (avg_r + 0.5) * cell_h
        else:
            target_cx = crop_w / 2
            target_cy = crop_h / 2

    full_x = x1 + target_cx
    full_y = y1 + target_cy

    M_full = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_full)
    pt_orig = M_inv @ np.array([full_x, full_y, 1.0])

    return ((pt_orig[0], pt_orig[1]), binary)


def compute_canonical_angle(shape_type, long_axis_angle, binary_grid):
    """
    计算方块相对于"正方向"的旋转角度。

    正方向定义 (所有方块在正方向时, "特征/凸起" 朝上):
      - L: 躺倒, 凸角朝上     ■         正方向时长边水平, 凸角在 top-left
                              ■ ■ ■
      - J: 躺倒, 凸角朝上         ■     正方向时长边水平, 凸角在 top-right
                              ■ ■ ■
      - T: 凸起朝上             ■       正方向时长边水平, 凸起在 top-center
                              ■ ■ ■
      - Z: 长边在底部          ■ ■       正方向时长边水平, 偏移在 top-left
                                ■ ■
      - N: 长边在底部            ■ ■     正方向时长边水平, 偏移在 top-right
                              ■ ■
      - I: 竖直, 向上          ■         正方向时长边竖直 (长边角度=90度)
                              ■
                              ■
      - O: 平放, 向上          ■ ■       正方向时边水平 (角度=0度)
                              ■ ■

    参数:
        shape_type: 方块类型
        long_axis_angle: minAreaRect 的长边角度 (已归一化到 [-90, 90)), 单位度
        binary_grid: 栅格化后的二值网格 (2x3, 1x4, 或 2x2)
    返回:
        rotation_deg: 相对于正方向的旋转角度, 范围 (-180, 180]
                      正值 = 顺时针旋转 (图像坐标系, y轴向下)
    """
    # long_axis_angle 是长边相对水平线的角度, 范围 [-90, 90)
    # 当方块旋转校正到 0 度后, 长边是水平的
    # binary_grid 是在校正后拍的栅格

    # ===== I 型: 正方向是竖直 (长边角度 = 90 度) =====
    if shape_type == 'I':
        # 正方向: 长边竖直 = 90 度
        # 当前长边角度 = long_axis_angle
        # 偏差 = long_axis_angle - 90
        rotation = long_axis_angle - 90.0
        # 归一化到 (-180, 180]
        while rotation > 180: rotation -= 360
        while rotation <= -180: rotation += 360
        return rotation

    # ===== O 型: 正方向是平放 (角度 = 0 度) =====
    if shape_type == 'O':
        # O 是正方形, 旋转 90 度等价, 所以归一化到 (-45, 45]
        rotation = long_axis_angle
        while rotation > 45: rotation -= 90
        while rotation <= -45: rotation += 90
        return rotation

    # ===== L, J, T, Z, N 型: 正方向是长边水平, 特征朝上 =====
    if binary_grid is None:
        return long_axis_angle

    rows, cols = binary_grid.shape

    # 判断特征在栅格中的位置 (top/bottom/left/right)
    # 校正后: row 0 = 上方, row 1 = 下方, col 0 = 左, col 2 = 右
    top_count = int(np.sum(binary_grid[0]))
    bottom_count = int(np.sum(binary_grid[1]))

    # 定义每种方块在"正方向"时的正确栅格模式:
    #
    # L 正方向: [[1,0,0],[1,1,1]]  -> top=1, bottom=3, 特征(单块)在 top-left
    # J 正方向: [[0,0,1],[1,1,1]]  -> top=1, bottom=3, 特征(单块)在 top-right
    # T 正方向: [[0,1,0],[1,1,1]]  -> top=1, bottom=3, 特征(单块)在 top-center
    # Z 正方向: [[1,1,0],[0,1,1]]  -> top=2, bottom=2, 左上 + 右下
    # N 正方向: [[0,1,1],[1,1,0]]  -> top=2, bottom=2, 右上 + 左下

    # 额外角度补偿: 如果特征朝下(而非朝上), 需要加 180 度
    feature_flip = 0.0

    if shape_type in ['L', 'J', 'T']:
        # 正方向: 单块(特征)在 top (top_count=1, bottom_count=3)
        if top_count == 1 and bottom_count == 3:
            # 特征已在上方, 无需翻转
            feature_flip = 0.0
        elif top_count == 3 and bottom_count == 1:
            # 特征在下方, 需要翻转 180 度
            feature_flip = 180.0
        elif top_count == 2 and bottom_count == 2:
            # 方块处于 "侧躺" 状态 (如 T 型凸起朝左或朝右)
            # 此时长边是竖直的, 但 minAreaRect 已经把角度归到了 [-90,90)
            # 检查左右分布
            left_count = int(np.sum(binary_grid[:, 0]))
            right_count = int(np.sum(binary_grid[:, -1]))
            if left_count == 1:
                # 特征在左 -> 相对正方向(朝上)需要顺时针转 90 度
                feature_flip = -90.0
            elif right_count == 1:
                # 特征在右 -> 相对正方向(朝上)需要逆时针转 90 度
                feature_flip = 90.0

    elif shape_type == 'Z':
        # Z 正方向: [[1,1,0],[0,1,1]] -> top-left 有块, bottom-right 有块
        # 判断当前是正Z还是翻转Z
        if binary_grid[0, 0] == 1 and binary_grid[1, -1] == 1:
            # 正常 Z 布局
            feature_flip = 0.0
        elif binary_grid[0, -1] == 1 and binary_grid[1, 0] == 1:
            # 翻转了 180 度
            feature_flip = 180.0

    elif shape_type == 'N':
        # N 正方向: [[0,1,1],[1,1,0]] -> top-right 有块, bottom-left 有块
        if binary_grid[0, -1] == 1 and binary_grid[1, 0] == 1:
            # 正常 N 布局
            feature_flip = 0.0
        elif binary_grid[0, 0] == 1 and binary_grid[1, -1] == 1:
            # 翻转了 180 度
            feature_flip = 180.0

    # 最终角度 = 长边角度 + 特征翻转补偿
    # long_axis_angle = 0 表示长边水平, 正方向也是长边水平(I除外)
    rotation = long_axis_angle + feature_flip

    # 归一化到 (-180, 180]
    while rotation > 180:
        rotation -= 360
    while rotation <= -180:
        rotation += 360

    return rotation


def detect_tetris_pieces(image_path, debug=True):
    """
    主检测函数。
    对每个识别到的方块, 根据颜色判断类型, 然后通过栅格化分析法定位:
      - L/J: 转角块中心 (栅格中有2个垂直邻居的格子)
      - T:   中间块中心 (栅格中有3个邻居的格子)
      - Z/N/I/O: 几何中心 (所有被占用格子的平均中心)
    返回列表: [{'type', 'center_px', 'angle', 'contour'}, ...]
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    results = []
    debug_img = img.copy() if debug else None

    px_per_mm = None  # 像素/mm 比例, 用 I 型标定

    for color_name, ranges in COLOR_RANGES.items():
        # --- 1. 颜色分割 ---
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

        # --- 2. 形态学闭运算 (弥合 2mm 缝隙) ---
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # --- 3. 查找轮廓 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000:
                continue

            # --- 4. minAreaRect 获取角度和几何信息 ---
            rect = cv2.minAreaRect(cnt)
            center, (rw, rh), angle = rect

            # 确保 rw >= rh (长边 >= 短边)
            if rw < rh:
                rw, rh = rh, rw
                angle += 90

            # 归一化角度到 [-90, 90)
            while angle > 90:
                angle -= 180
            while angle <= -90:
                angle += 180

            # solidity (填充率)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # --- 5. 几何粗分类, 验证颜色类型 ---
            geo_class = classify_by_geometry(rw, rh, solidity)
            if geo_class == 'I' and color_name != 'I':
                continue
            if geo_class == 'O' and color_name != 'O':
                continue

            shape_type = color_name

            # --- 6. 计算目标中心点 (栅格化分析法) ---
            (target_x, target_y), binary_grid = find_target_center_by_grid(
                cnt, shape_type, mask.shape
            )

            # --- 7. 计算相对于正方向的旋转角度 ---
            canonical_angle = compute_canonical_angle(shape_type, angle, binary_grid)

            # --- 8. 标定: 用 I 型计算 px_per_mm ---
            if shape_type == 'I' and px_per_mm is None:
                i_long_mm = 4 * BLOCK_SIZE_MM + 3 * GAP_MM  # 78mm
                px_per_mm = rw / i_long_mm
                print(f"[Calibration] I-piece: {rw:.1f}px / {i_long_mm}mm = {px_per_mm:.3f} px/mm")

            result = {
                'type': shape_type,
                'center_px': (round(target_x, 1), round(target_y, 1)),
                'angle': round(canonical_angle, 2),
                'rect_size_px': (round(rw, 1), round(rh, 1)),
                'contour': cnt,
            }
            results.append(result)

            # --- 9. Debug 可视化 ---
            if debug_img is not None:
                tx, ty = int(target_x), int(target_y)

                # 绘制轮廓 (绿色)
                cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 2)

                # 绘制 minAreaRect (青色)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(debug_img, [box], -1, (255, 255, 0), 2)

                # 绘制目标吸取中心 (红色实心圆 + 白色边框)
                cv2.circle(debug_img, (tx, ty), 8, (0, 0, 255), -1)
                cv2.circle(debug_img, (tx, ty), 10, (255, 255, 255), 2)

                # 绘制正方向箭头 (黄色)
                # 正方向 = 0 度时指向图像"上方" (y轴负方向)
                # canonical_angle 表示方块相对于正方向旋转了多少度
                # 所以当前正方向在图像中指向: -90 + canonical_angle (从水平轴算起)
                # 换一种理解: 正方向在图像中的方位 = 从"向上"顺时针旋转 canonical_angle 度
                line_len = 80
                # "向上" = -90 度(从水平轴算), 再加上 canonical_angle
                arrow_angle_rad = math.radians(-90 + canonical_angle)
                ex = int(tx + line_len * math.cos(arrow_angle_rad))
                ey = int(ty + line_len * math.sin(arrow_angle_rad))
                cv2.arrowedLine(debug_img, (tx, ty), (ex, ey), (0, 0, 0), 2, tipLength=0.2)

                # 标注文字 (黑色描边 + 白色文字)
                label = f"{shape_type} {canonical_angle:.1f}deg"
                cv2.putText(debug_img, label, (tx + 15, ty - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(debug_img, label, (tx + 15, ty - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # =========================================================================
    # 输出结果
    # =========================================================================
    print(f"\n{'='*60}")
    print(f" Detection Results: {image_path}")
    print(f"{'='*60}")
    if px_per_mm:
        print(f" Scale: {px_per_mm:.3f} px/mm")
    print(f" Found {len(results)} pieces:")
    print(f"{'='*60}")

    for i, r in enumerate(results):
        cx, cy = r['center_px']
        ang = r['angle']
        rw, rh = r['rect_size_px']
        line = f" [{i+1}] {r['type']:>2s} | pixel=({cx:7.1f}, {cy:7.1f})"
        if px_per_mm:
            mx = cx / px_per_mm
            my = cy / px_per_mm
            line += f" | mm=({mx:7.1f}, {my:7.1f})"
        line += f" | angle={ang:7.2f} deg | size=({rw:.0f}x{rh:.0f})"
        print(line)

    print(f"{'='*60}\n")

    # 保存 debug 图
    if debug_img is not None:
        out_path = image_path.replace('.png', '_debug.png')
        cv2.imwrite(out_path, debug_img)
        print(f"[Debug] Saved to: {out_path}")

    return results


if __name__ == "__main__":
    images = sys.argv[1:] if len(sys.argv) > 1 else ['image_1.png']
    for img_path in images:
        detect_tetris_pieces(img_path, debug=True)
