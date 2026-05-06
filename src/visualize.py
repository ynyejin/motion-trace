import csv
import os
import numpy as np
import matplotlib.pyplot as plt

POSE_CSV_PATH = "output/pose_data.csv"
ENERGY_CSV_PATH = "output/energy_data.csv"
ANGLE_CSV_PATH = "output/angle_data.csv"

ENERGY_GRAPH_PATH = "output/energy_graph.png"
TRAJECTORY_3D_PATH = "output/trajectory_3d.png"
ANGLE_GRAPH_PATH = "output/angle_graph.png"
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") == "1"

os.makedirs("output", exist_ok=True)


def load_pose_data():
    pose_data = {}

    with open(POSE_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            joint = row["joint"]

            if joint not in pose_data:
                pose_data[joint] = {
                    "time": [],
                    "x": [],
                    "y": []
                }

            pose_data[joint]["time"].append(float(row["time"]))
            pose_data[joint]["x"].append(float(row["x"]))
            pose_data[joint]["y"].append(float(row["y"]))

    return pose_data


def load_energy_data():
    times = []
    total_energy = []

    with open(ENERGY_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            times.append(float(row["time"]))
            total_energy.append(float(row["total_energy"]))

    return times, total_energy


def load_angle_data():
    angle_data = {}

    with open(ANGLE_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            joint = row["joint"]

            if joint not in angle_data:
                angle_data[joint] = {
                    "time": [],
                    "angle": []
                }

            angle_data[joint]["time"].append(float(row["time"]))
            angle_data[joint]["angle"].append(float(row["angle"]))

    return angle_data


def plot_energy_graph():
    times, total_energy = load_energy_data()

    max_index = int(np.argmax(total_energy))
    max_time = times[max_index]
    max_energy = total_energy[max_index]

    plt.figure(figsize=(12, 6))
    plt.plot(times, total_energy, linewidth=2)
    plt.scatter(max_time, max_energy, s=120)
    plt.axvline(max_time, linestyle="--")

    plt.title("DanceForge Movement Energy Graph")
    plt.xlabel("Time (sec)")
    plt.ylabel("Movement Energy")
    plt.text(
        max_time,
        max_energy,
        f"Killing Part\n{max_time:.2f}s",
        fontsize=10
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ENERGY_GRAPH_PATH, dpi=300)

    print(f"에너지 그래프 저장: {ENERGY_GRAPH_PATH}")
    print(f"킬링파트: {max_time:.2f}초 / 에너지 {max_energy:.2f}")


def plot_3d_trajectory():
    pose_data = load_pose_data()

    target_joints = [
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
        "left_shoulder",
        "right_shoulder"
    ]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    for joint in target_joints:
        if joint not in pose_data:
            continue

        x = pose_data[joint]["x"]
        y = pose_data[joint]["y"]
        t = pose_data[joint]["time"]

        ax.plot(x, y, t, label=joint)

    ax.set_title("DanceForge 3D Joint Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_zlabel("Time (sec)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(TRAJECTORY_3D_PATH, dpi=300)

    print(f"3D 궤적 그래프 저장: {TRAJECTORY_3D_PATH}")


def plot_angle_graph():
    angle_data = load_angle_data()

    plt.figure(figsize=(12, 6))

    for joint, data in angle_data.items():
        plt.plot(data["time"], data["angle"], linewidth=2, label=joint)

    plt.title("DanceForge Joint Angle Graph")
    plt.xlabel("Time (sec)")
    plt.ylabel("Angle (degree)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ANGLE_GRAPH_PATH, dpi=300)

    print(f"관절 각도 그래프 저장: {ANGLE_GRAPH_PATH}")


def main():
    plot_energy_graph()
    plot_3d_trajectory()
    plot_angle_graph()

    print("Phase 4 시각화 완료!")
    print("그래프 창을 닫으면 프로그램이 종료됩니다.")

    if SHOW_PLOTS:
        plt.show()

def run_visualization():
    plot_energy_graph()
    plot_3d_trajectory()
    plot_angle_graph()


if __name__ == "__main__":
    main()


