import cv2
import mediapipe as mp
import numpy as np
import os
import csv

VIDEO_PATH = os.environ.get("VIDEO_PATH", "data/test.mp4")
SHOW_WINDOW = os.environ.get("SHOW_WINDOW", "1") == "1"

OUTPUT_VIDEO_PATH = "output/avatar_side_by_side.mp4"
POSE_CSV_PATH = "output/pose_data.csv"
ENERGY_CSV_PATH = "output/energy_data.csv"
ANGLE_CSV_PATH = "output/angle_data.csv"
REPORT_PATH = "output/summary_report.txt"

os.makedirs("output", exist_ok=True)

mp_pose = mp.solutions.pose

LANDMARKS = {
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}

UPPER_BODY = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist"
]

LOWER_BODY = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

UPPER_BODY_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
]

LOWER_BODY_CONNECTIONS = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

UPPER_COLOR = (255, 0, 0)
LOWER_COLOR = (0, 0, 255)
POINT_COLOR = (0, 255, 255)

SMOOTHING_ALPHA = 0.6
JUMP_THRESHOLD = 120


def extract_keypoints(results, width, height):
    keypoints = {}

    if not results.pose_landmarks:
        return keypoints

    for name, landmark_id in LANDMARKS.items():
        lm = results.pose_landmarks.landmark[landmark_id]
        x = int(lm.x * width)
        y = int(lm.y * height)
        keypoints[name] = (x, y)

    return keypoints


def draw_thick_segment(img, p1, p2, thickness_start, thickness_end, color):
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length == 0:
        return

    direction = direction / length
    perpendicular = np.array([-direction[1], direction[0]])

    p1_offset = perpendicular * (thickness_start / 2)
    p2_offset = perpendicular * (thickness_end / 2)

    polygon = np.array([
        p1 + p1_offset,
        p2 + p2_offset,
        p2 - p2_offset,
        p1 - p1_offset
    ], dtype=np.int32)

    cv2.fillPoly(img, [polygon], color, lineType=cv2.LINE_AA)

    cv2.circle(img, tuple(p1.astype(int)), int(thickness_start / 2), color, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, tuple(p2.astype(int)), int(thickness_end / 2), color, -1, lineType=cv2.LINE_AA)


def draw_glow(img, mask_color=(80, 180, 255)):
    glow = cv2.GaussianBlur(img, (0, 0), 10)
    return cv2.addWeighted(glow, 0.35, img, 1.0, 0)


def draw_avatar(frame, keypoints):
    avatar = np.zeros_like(frame)
    avatar[:] = (8, 8, 12)

    body_color = (70, 190, 255)
    arm_color = (255, 145, 90)
    leg_color = (110, 135, 255)
    head_color = (230, 230, 235)
    outline_color = (255, 255, 255)
    joint_color = (255, 255, 255)

    # 몸통: 어깨~골반을 채운 실루엣 형태
    torso_joints = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]

    if all(j in keypoints for j in torso_joints):
        ls = np.array(keypoints["left_shoulder"])
        rs = np.array(keypoints["right_shoulder"])
        rh = np.array(keypoints["right_hip"])
        lh = np.array(keypoints["left_hip"])

        shoulder_width = np.linalg.norm(rs - ls)
        hip_width = np.linalg.norm(rh - lh)

        shoulder_expand = max(6, shoulder_width * 0.08)
        hip_expand = max(5, hip_width * 0.06)

        shoulder_dir = rs - ls
        hip_dir = rh - lh

        if np.linalg.norm(shoulder_dir) != 0:
            shoulder_dir = shoulder_dir / np.linalg.norm(shoulder_dir)
        if np.linalg.norm(hip_dir) != 0:
            hip_dir = hip_dir / np.linalg.norm(hip_dir)

        torso = np.array([
            ls - shoulder_dir * shoulder_expand,
            rs + shoulder_dir * shoulder_expand,
            rh + hip_dir * hip_expand,
            lh - hip_dir * hip_expand
        ], dtype=np.int32)

        cv2.fillPoly(avatar, [torso], body_color, lineType=cv2.LINE_AA)
        cv2.polylines(avatar, [torso], True, outline_color, 2, lineType=cv2.LINE_AA)

    # 팔: 위팔은 두껍게, 아래팔은 조금 얇게
    arm_segments = [
        ("left_shoulder", "left_elbow", 24, 20),
        ("left_elbow", "left_wrist", 20, 14),
        ("right_shoulder", "right_elbow", 24, 20),
        ("right_elbow", "right_wrist", 20, 14),
    ]

    for start, end, t1, t2 in arm_segments:
        if start in keypoints and end in keypoints:
            draw_thick_segment(avatar, keypoints[start], keypoints[end], t1, t2, arm_color)

    # 다리: 허벅지는 두껍게, 종아리는 조금 얇게
    leg_segments = [
        ("left_hip", "left_knee", 30, 24),
        ("left_knee", "left_ankle", 24, 16),
        ("right_hip", "right_knee", 30, 24),
        ("right_knee", "right_ankle", 24, 16),
    ]

    for start, end, t1, t2 in leg_segments:
        if start in keypoints and end in keypoints:
            draw_thick_segment(avatar, keypoints[start], keypoints[end], t1, t2, leg_color)

    # 골반 연결
    if "left_hip" in keypoints and "right_hip" in keypoints:
        draw_thick_segment(
            avatar,
            keypoints["left_hip"],
            keypoints["right_hip"],
            22,
            22,
            body_color
        )

    # 목 + 머리
    if "left_shoulder" in keypoints and "right_shoulder" in keypoints:
        ls = np.array(keypoints["left_shoulder"])
        rs = np.array(keypoints["right_shoulder"])

        shoulder_center = ((ls + rs) / 2).astype(int)
        shoulder_width = np.linalg.norm(ls - rs)

        head_radius = int(max(18, shoulder_width * 0.25))
        neck_top = (
            int(shoulder_center[0]),
            int(shoulder_center[1] - shoulder_width * 0.30)
        )
        head_center = (
            int(shoulder_center[0]),
            int(shoulder_center[1] - shoulder_width * 0.72)
        )

        cv2.line(avatar, tuple(shoulder_center), neck_top, body_color, 14, lineType=cv2.LINE_AA)
        cv2.circle(avatar, head_center, head_radius, head_color, -1, lineType=cv2.LINE_AA)
        cv2.circle(avatar, head_center, head_radius, outline_color, 2, lineType=cv2.LINE_AA)

    # 관절 포인트는 작게만 표시
    for point in keypoints.values():
        cv2.circle(avatar, point, 4, joint_color, -1, lineType=cv2.LINE_AA)

    avatar = draw_glow(avatar)

    return avatar


def draw_overlay(frame, keypoints):
    overlay = frame.copy()

    for start, end in UPPER_BODY_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(overlay, keypoints[start], keypoints[end], UPPER_COLOR, 4)

    for start, end in LOWER_BODY_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(overlay, keypoints[start], keypoints[end], LOWER_COLOR, 4)

    for point in keypoints.values():
        cv2.circle(overlay, point, 6, POINT_COLOR, -1)

    return overlay


def calculate_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def smooth_keypoints(current_keypoints, previous_keypoints):
    if previous_keypoints is None:
        return current_keypoints

    smoothed = {}

    for joint, current_point in current_keypoints.items():
        if joint not in previous_keypoints:
            smoothed[joint] = current_point
            continue

        prev_point = previous_keypoints[joint]
        distance = calculate_distance(prev_point, current_point)

        if distance > JUMP_THRESHOLD:
            x = int(prev_point[0] * 0.8 + current_point[0] * 0.2)
            y = int(prev_point[1] * 0.8 + current_point[1] * 0.2)
        else:
            x = int(prev_point[0] * (1 - SMOOTHING_ALPHA) + current_point[0] * SMOOTHING_ALPHA)
            y = int(prev_point[1] * (1 - SMOOTHING_ALPHA) + current_point[1] * SMOOTHING_ALPHA)

        smoothed[joint] = (x, y)

    for joint, prev_point in previous_keypoints.items():
        if joint not in smoothed:
            smoothed[joint] = prev_point

    return smoothed


def calculate_energy(current_keypoints, previous_keypoints, target_joints):
    energy = 0.0

    if previous_keypoints is None:
        return energy

    for joint in target_joints:
        if joint in current_keypoints and joint in previous_keypoints:
            energy += calculate_distance(previous_keypoints[joint], current_keypoints[joint])

    return energy


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cosine_angle))


def calculate_joint_angles(keypoints):
    angles = {}

    if all(j in keypoints for j in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_elbow"] = calculate_angle(
            keypoints["left_shoulder"],
            keypoints["left_elbow"],
            keypoints["left_wrist"]
        )

    if all(j in keypoints for j in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_elbow"] = calculate_angle(
            keypoints["right_shoulder"],
            keypoints["right_elbow"],
            keypoints["right_wrist"]
        )

    if all(j in keypoints for j in ["left_hip", "left_knee", "left_ankle"]):
        angles["left_knee"] = calculate_angle(
            keypoints["left_hip"],
            keypoints["left_knee"],
            keypoints["left_ankle"]
        )

    if all(j in keypoints for j in ["right_hip", "right_knee", "right_ankle"]):
        angles["right_knee"] = calculate_angle(
            keypoints["right_hip"],
            keypoints["right_knee"],
            keypoints["right_ankle"]
        )

    return angles


def save_pose_csv(pose_rows):
    with open(POSE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "joint", "x", "y"])
        writer.writerows(pose_rows)


def save_energy_csv(energy_rows):
    with open(ENERGY_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "upper_energy", "lower_energy", "total_energy"])
        writer.writerows(energy_rows)


def save_angle_csv(angle_rows):
    with open(ANGLE_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time", "joint", "angle"])
        writer.writerows(angle_rows)


def save_report(energy_rows, angle_rows):
    if not energy_rows:
        return

    total_upper = sum(row[2] for row in energy_rows)
    total_lower = sum(row[3] for row in energy_rows)
    total_energy = sum(row[4] for row in energy_rows)

    if total_energy == 0:
        upper_ratio = 0
        lower_ratio = 0
    else:
        upper_ratio = total_upper / total_energy * 100
        lower_ratio = total_lower / total_energy * 100

    max_energy_row = max(energy_rows, key=lambda row: row[4])
    max_frame = max_energy_row[0]
    max_time = max_energy_row[1]
    max_energy = max_energy_row[4]

    if upper_ratio > lower_ratio:
        main_body_part = "상체 중심"
    elif lower_ratio > upper_ratio:
        main_body_part = "하체 중심"
    else:
        main_body_part = "상체와 하체가 균형적인"

    angle_summary = {}

    for row in angle_rows:
        joint = row[2]
        angle = row[3]

        if joint not in angle_summary:
            angle_summary[joint] = []

        angle_summary[joint].append(angle)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("DanceForge Motion Analysis Report\n")
        f.write("=================================\n\n")

        f.write("[프로젝트 요약]\n")
        f.write("입력된 안무 영상에서 MediaPipe Pose를 이용해 주요 관절을 추출하고,\n")
        f.write("프레임별 관절 좌표, 움직임 에너지, 관절 각도를 기반으로 안무의 움직임 패턴을 분석하였다.\n\n")

        f.write("[전처리 및 안정화]\n")
        f.write("관절 좌표가 순간적으로 튀는 현상을 줄이기 위해 이전 프레임 좌표를 활용한 smoothing 처리를 적용하였다.\n")
        f.write("특정 관절이 일시적으로 검출되지 않는 경우에는 이전 프레임의 좌표를 사용하여 보간하였다.\n\n")

        f.write("[움직임 에너지 분석]\n")
        f.write(f"- 전체 움직임 에너지: {total_energy:.2f}\n")
        f.write(f"- 상체 움직임 비중: {upper_ratio:.2f}%\n")
        f.write(f"- 하체 움직임 비중: {lower_ratio:.2f}%\n")
        f.write(f"- 분석 결과, 이 안무는 {main_body_part} 움직임이 두드러진다.\n\n")

        f.write("[킬링파트 자동 감지]\n")
        f.write(f"- 가장 움직임이 큰 구간은 {max_time:.2f}초 지점이다.\n")
        f.write(f"- 해당 프레임 번호: {max_frame}\n")
        f.write(f"- 해당 구간 에너지 값: {max_energy:.2f}\n\n")

        f.write("[관절 각도 요약]\n")

        for joint, angle_list in angle_summary.items():
            avg_angle = np.mean(angle_list)
            min_angle = np.min(angle_list)
            max_angle = np.max(angle_list)

            f.write(f"- {joint}\n")
            f.write(f"  평균 각도: {avg_angle:.2f}도\n")
            f.write(f"  최소 각도: {min_angle:.2f}도\n")
            f.write(f"  최대 각도: {max_angle:.2f}도\n")

        f.write("\n[결론]\n")
        f.write("본 시스템은 안무 영상에서 사람의 주요 관절을 추출하고,\n")
        f.write("2D 실루엣 아바타, 움직임 에너지, 관절 각도, 3D 궤적 그래프를 통해\n")
        f.write("안무의 동작 특성을 시각적이고 정량적으로 분석할 수 있다.\n")


def run_analysis():
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("영상 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width * 2, height)
    )

    pose_rows = []
    energy_rows = []
    angle_rows = []

    previous_keypoints = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        time_sec = frame_idx / fps

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        keypoints = extract_keypoints(results, width, height)
        keypoints = smooth_keypoints(keypoints, previous_keypoints)

        for joint, (x, y) in keypoints.items():
            pose_rows.append([frame_idx, time_sec, joint, x, y])

        upper_energy = calculate_energy(keypoints, previous_keypoints, UPPER_BODY)
        lower_energy = calculate_energy(keypoints, previous_keypoints, LOWER_BODY)
        total_energy = upper_energy + lower_energy

        energy_rows.append([
            frame_idx,
            time_sec,
            upper_energy,
            lower_energy,
            total_energy
        ])

        angles = calculate_joint_angles(keypoints)

        for joint, angle in angles.items():
            angle_rows.append([
                frame_idx,
                time_sec,
                joint,
                angle
            ])

        overlay_frame = draw_overlay(frame, keypoints)
        avatar_frame = draw_avatar(frame, keypoints)
        combined = np.hstack((overlay_frame, avatar_frame))

        out.write(combined)

        if SHOW_WINDOW:
            cv2.imshow("DanceForge - Motion Analysis", combined)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        previous_keypoints = keypoints.copy()
        frame_idx += 1

    cap.release()
    out.release()
    pose.close()

    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    save_pose_csv(pose_rows)
    save_energy_csv(energy_rows)
    save_angle_csv(angle_rows)
    save_report(energy_rows, angle_rows)

    print("DanceForge 분석 완료!")
    print(f"영상 저장: {OUTPUT_VIDEO_PATH}")
    print(f"관절 좌표 저장: {POSE_CSV_PATH}")
    print(f"에너지 데이터 저장: {ENERGY_CSV_PATH}")
    print(f"각도 데이터 저장: {ANGLE_CSV_PATH}")
    print(f"분석 리포트 저장: {REPORT_PATH}")


if __name__ == "__main__":
    run_analysis()