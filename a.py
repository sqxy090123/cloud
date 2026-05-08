import math

def reward_function(params):
    # ======================
    # 基础参数
    # ======================
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    steering_abs = abs(params['steering_angle'])
    steering_raw = params['steering_angle']
    heading = params['heading']
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    all_wheels_on_track = params['all_wheels_on_track']
    is_reversed = params['is_reversed']
    progress = params['progress']
    steps = params['steps']
    x = params['x']
    y = params['y']

    # ======================
    # 安全底线
    # ======================
    if not all_wheels_on_track or is_reversed:
        return -15.0

    half_width = track_width / 2.0

    if distance_from_center > half_width:
        return -15.0

    if progress < 1.0 and steps > 150:
        return -5.0

    # ======================
    # 速度裁剪
    # ======================
    speed = max(0.5, min(speed, 4.0))

    # ======================
    # 弯道检测（平滑曲率）
    # ======================
    prev_idx, next_idx = closest_waypoints

    prev_pt = waypoints[prev_idx]
    next_pt = waypoints[next_idx]

    track_dir = math.degrees(math.atan2(
        next_pt[1] - prev_pt[1],
        next_pt[0] - prev_pt[0]
    ))

    idx1 = min(next_idx + 3, len(waypoints) - 1)
    idx2 = min(next_idx + 4, len(waypoints) - 1)

    pt1 = waypoints[idx1]
    pt2 = waypoints[idx2]

    dir1 = math.degrees(math.atan2(
        pt1[1] - prev_pt[1],
        pt1[0] - prev_pt[0]
    ))

    dir2 = math.degrees(math.atan2(
        pt2[1] - prev_pt[1],
        pt2[0] - prev_pt[0]
    ))

    curve1 = abs(dir1 - track_dir)
    curve2 = abs(dir2 - track_dir)

    if curve1 > 180:
        curve1 = 360 - curve1

    if curve2 > 180:
        curve2 = 360 - curve2

    curve = (curve1 + curve2) / 2.0

    # ======================
    # 弯道等级
    # ======================
    if curve > 25:
        curve_intensity = 2
    elif curve > 12:
        curve_intensity = 1
    else:
        curve_intensity = 0

    # ======================
    # 动态速度目标
    # ======================
    if curve_intensity == 2:
        target_speed = 2.5
    elif curve_intensity == 1:
        target_speed = 3.0
    else:
        target_speed = 3.8

    # ======================
    # 速度奖励
    # ======================
    speed_ratio = min(1.0, speed / target_speed)

    speed_reward = speed_ratio ** 1.5

    overspeed_reward = 0.0

    if speed > target_speed:
        overspeed = min(speed - target_speed, 0.3)
        overspeed_reward = overspeed * 0.5

    # ======================
    # 航向对齐
    # ======================
    dir_diff = abs(track_dir - heading)

    if dir_diff > 180:
        dir_diff = 360 - dir_diff

    dir_rad = math.radians(dir_diff)

    heading_reward = (
        (math.cos(dir_rad) + 1.0) / 2.0
    ) ** 1.8

    # ======================
    # 放宽后的中心奖励
    # 核心修改：
    # sigma 从 0.3 -> 0.42
    # 并降低整体权重
    # ======================
    distance_ratio = distance_from_center / half_width

    center_reward = math.exp(
        -0.5 * (distance_ratio / 0.42) ** 2
    )

    # 极限边缘才明显惩罚
    if distance_ratio > 0.95:
        center_reward *= 0.75

    center_reward = max(0.35, min(1.0, center_reward))

    # ======================
    # 转向控制（柔性）
    # ======================
    if curve_intensity == 0:
        desired_steer = 0.0
    else:
        desired_steer = min(
            25.0,
            curve * 0.3 + speed * 0.7
        )

    steer_diff = abs(
        steering_abs - desired_steer
    )

    turn_penalty = (
        1.0 - min(1.0, steer_diff / 20.0) * 0.3
    )

    turn_penalty = max(
        0.5,
        min(1.0, turn_penalty)
    )

    turn_bonus = (turn_penalty - 0.7) * 0.2

    # ======================
    # 直道蛇形惩罚
    # ======================
    if curve_intensity == 0:

        if speed > 3.5 and steering_abs > 10:
            turn_bonus -= 0.12

        elif steering_abs > 15:
            turn_bonus -= 0.08

        elif steering_abs > 12:
            turn_bonus -= 0.04

    # ======================
    # 左偏纠正
    # ======================
    left_bias_penalty = 0.0

    signed_diff = track_dir - heading

    if signed_diff > 180:
        signed_diff -= 360

    elif signed_diff < -180:
        signed_diff += 360

    if curve_intensity == 0:

        if signed_diff > 8 and steering_raw > 4:
            left_bias_penalty = -0.08

        elif signed_diff < -8 and steering_raw < -4:
            left_bias_penalty = -0.08

    # ======================
    # 进度奖励
    # ======================
    progress_reward = progress * 0.012

    step_penalty = -0.012

    completion_bonus = 6.0 if progress >= 99.9 else 0.0

    # ======================
    # 动态未来预瞄
    # ======================
    if speed > 3.4:
        idx_list = [
            min(next_idx + i, len(waypoints) - 1)
            for i in [5, 7, 9]
        ]
    else:
        idx_list = [
            min(next_idx + i, len(waypoints) - 1)
            for i in [3, 4, 5]
        ]

    future_diffs = []

    for idx in idx_list:

        pt = waypoints[idx]

        dir_val = math.degrees(math.atan2(
            pt[1] - y,
            pt[0] - x
        ))

        diff_val = abs(dir_val - heading)

        if diff_val > 180:
            diff_val = 360 - diff_val

        future_diffs.append(diff_val)

    avg_diff_future = (
        sum(future_diffs) / len(future_diffs)
    )

    future_reward = (
        (
            (math.cos(math.radians(avg_diff_future)) + 1.0)
            / 2.0
        ) ** 1.5
    )

    future_reward *= 0.20

    # ======================
    # 基础组合奖励
    # ======================
    reward = (
        speed_reward * 0.35 +
        overspeed_reward +
        heading_reward * 0.20 +
        center_reward * 0.08 +   # 原来 0.15 -> 0.08
        turn_bonus * 0.10 +
        progress_reward +
        step_penalty +
        completion_bonus +
        future_reward +
        left_bias_penalty
    )

    # ======================
    # 大拐平滑惩罚
    # 放宽触发阈值
    # ======================
    if dir_diff > 32 and curve_intensity > 0:

        factor = max(
            0.45,
            1.0 - (dir_diff - 32) * 0.015
        )

        reward *= factor

    # ======================
    # 方向一致性
    # ======================
    signed_dir = heading - track_dir

    if signed_dir > 180:
        signed_dir -= 360

    elif signed_dir < -180:
        signed_dir += 360

    if curve_intensity == 0:

        if signed_dir * steering_raw > 0:
            reward -= 0.12

    else:

        if signed_dir * steering_raw > 0:
            reward -= 0.08

    # ======================
    # 防掉头
    # ======================
    if (
        dir_diff > 80 and
        speed < 1.5 and
        steering_abs > 20
    ):
        reward -= 1.0

    # ======================
    # 裁剪
    # ======================
    reward = max(-15.0, min(reward, 15.0))

    return float(reward)