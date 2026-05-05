import cv2
import mediapipe as mp
import numpy as np
import os
import csv

VIDEO_PATH = "data/test.mp4"
OUTPUT_VIDEO_PATH = "output/avatar_side_by_side.mp4"
POSE_CSV_PATH = "output/pose_data.csv"
ENERGY_CSV_PATH = "output/energy_data.csv"
ANGLE_CSV_PATH = "output/angle_data.csv"
REPORT_PATH = "output/summary_report.txt"

os.makedirs("output", exist_ok=True)

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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


def draw_avatar(frame, keypoints):
    avatar = np.zeros_like(frame)

    for start, end in UPPER_BODY_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(avatar, keypoints[start], keypoints[end], UPPER_COLOR, 4)

    for start, end in LOWER_BODY_CONNECTIONS:
        if start in keypoints and end in keypoints:
            cv2.line(avatar, keypoints[start], keypoints[end], LOWER_COLOR, 4)

    for point in keypoints.values():
        cv2.circle(avatar, point, 6, POINT_COLOR, -1)

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

        f.write("Energy Summary\n")
        f.write("--------------\n")
        f.write(f"Total Energy: {total_energy:.2f}\n")
        f.write(f"Upper Body Energy Ratio: {upper_ratio:.2f}%\n")
        f.write(f"Lower Body Energy Ratio: {lower_ratio:.2f}%\n\n")

        f.write("Highest Energy Moment\n")
        f.write("---------------------\n")
        f.write(f"Frame: {max_frame}\n")
        f.write(f"Time: {max_time:.2f} sec\n")
        f.write(f"Energy: {max_energy:.2f}\n\n")

        f.write("Joint Angle Summary\n")
        f.write("-------------------\n")

        for joint, angle_list in angle_summary.items():
            avg_angle = np.mean(angle_list)
            min_angle = np.min(angle_list)
            max_angle = np.max(angle_list)

            f.write(f"{joint}\n")
            f.write(f"  Average Angle: {avg_angle:.2f} degrees\n")
            f.write(f"  Min Angle: {min_angle:.2f} degrees\n")
            f.write(f"  Max Angle: {max_angle:.2f} degrees\n\n")


cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("영상 파일을 열 수 없습니다.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

    cv2.imshow("DanceForge - Phase 3 Motion Analysis", combined)
    out.write(combined)

    previous_keypoints = keypoints.copy()
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
pose.close()
cv2.destroyAllWindows()

save_pose_csv(pose_rows)
save_energy_csv(energy_rows)
save_angle_csv(angle_rows)
save_report(energy_rows, angle_rows)

print("Phase 3 완료!")
print(f"영상 저장: {OUTPUT_VIDEO_PATH}")
print(f"관절 좌표 저장: {POSE_CSV_PATH}")
print(f"에너지 데이터 저장: {ENERGY_CSV_PATH}")
print(f"각도 데이터 저장: {ANGLE_CSV_PATH}")
print(f"분석 리포트 저장: {REPORT_PATH}")