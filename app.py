import streamlit as st
import os
import subprocess
import sys

st.set_page_config(
    page_title="DanceForge",
    page_icon="🕺",
    layout="wide"
)

DATA_DIR = "data"
OUTPUT_DIR = "output"

INPUT_VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")

RAW_VIDEO_PATH = os.path.join(OUTPUT_DIR, "avatar_side_by_side.mp4")
VIDEO_PATH = os.path.join(OUTPUT_DIR, "avatar_side_by_side_web.mp4")

ENERGY_GRAPH_PATH = os.path.join(OUTPUT_DIR, "energy_graph.png")
TRAJECTORY_3D_PATH = os.path.join(OUTPUT_DIR, "trajectory_3d.png")
ANGLE_GRAPH_PATH = os.path.join(OUTPUT_DIR, "angle_graph.png")
REPORT_PATH = os.path.join(OUTPUT_DIR, "summary_report.txt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.title("🕺 DanceForge")
st.subheader("Idol Dance Motion Analysis & 3D Avatar System")

st.markdown(
    """
아이돌 안무 영상을 업로드하면 사람의 주요 관절을 추출하고,  
2D 아바타 / 움직임 에너지 / 관절 각도 / 3D 궤적을 한 페이지에서 확인할 수 있습니다.
"""
)

st.divider()

uploaded_file = st.file_uploader(
    "분석할 안무 영상을 업로드하세요",
    type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    with open(INPUT_VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.read())

    st.success("영상 업로드 완료")
    st.video(INPUT_VIDEO_PATH)

    if st.button("DanceForge 분석 시작"):
        # 이전 결과 삭제
        for filename in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, filename)
            if os.path.isfile(path):
                os.remove(path)

        env = os.environ.copy()
        env["VIDEO_PATH"] = INPUT_VIDEO_PATH
        env["SHOW_WINDOW"] = "0"
        env["SHOW_PLOTS"] = "0"

        with st.spinner("1단계: 영상 분석 중..."):
            result_main = subprocess.run(
                [sys.executable, "src/main.py"],
                capture_output=True,
                text=True,
                env=env
            )

        if result_main.returncode != 0:
            st.error("main.py 실행 중 오류가 발생했습니다.")
            st.code(result_main.stderr)
            st.stop()

        with st.spinner("2단계: 웹 재생용 영상 변환 중..."):
            result_ffmpeg = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", RAW_VIDEO_PATH,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    VIDEO_PATH
                ],
                capture_output=True,
                text=True
            )

        if result_ffmpeg.returncode != 0:
            st.error("ffmpeg 영상 변환 중 오류가 발생했습니다.")
            st.code(result_ffmpeg.stderr)
            st.stop()

        with st.spinner("3단계: 그래프 생성 중..."):
            result_visualize = subprocess.run(
                [sys.executable, "src/visualize.py"],
                capture_output=True,
                text=True,
                env=env
            )

        if result_visualize.returncode != 0:
            st.error("visualize.py 실행 중 오류가 발생했습니다.")
            st.code(result_visualize.stderr)
            st.stop()

        st.session_state["analysis_done"] = True
        st.success("분석 완료!")

else:
    st.info("먼저 영상을 업로드하세요.")

required_files = [
    VIDEO_PATH,
    ENERGY_GRAPH_PATH,
    TRAJECTORY_3D_PATH,
    ANGLE_GRAPH_PATH,
    REPORT_PATH
]

if not st.session_state["analysis_done"]:
    st.warning("아직 분석 결과가 생성되지 않았습니다. 영상을 업로드한 뒤 분석 시작 버튼을 눌러주세요.")
    st.stop()

missing_files = [path for path in required_files if not os.path.exists(path)]

if missing_files:
    st.error("분석은 완료됐지만 일부 결과 파일이 없습니다.")
    for path in missing_files:
        st.write(f"- {path}")
    st.stop()

st.divider()

st.header("1. 원본 + 2D 아바타 영상")

with open(VIDEO_PATH, "rb") as video_file:
    video_bytes = video_file.read()

st.video(video_bytes)

st.divider()

st.header("2. 움직임 분석 그래프")

col1, col2 = st.columns(2)

with col1:
    st.subheader("움직임 에너지 그래프")
    st.image(ENERGY_GRAPH_PATH, use_container_width=True)

with col2:
    st.subheader("관절 각도 그래프")
    st.image(ANGLE_GRAPH_PATH, use_container_width=True)

st.subheader("3D 관절 궤적 그래프")
st.image(TRAJECTORY_3D_PATH, use_container_width=True)

st.divider()

st.header("3. 분석 리포트")

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report_text = f.read()

st.text_area(
    "DanceForge Motion Analysis Report",
    value=report_text,
    height=450
)

st.divider()

st.header("4. 프로젝트 파이프라인")

st.markdown(
    """
```text
영상 업로드
    ↓
OpenCV 영상 처리
    ↓
MediaPipe Pose 관절 추출
    ↓
2D 아바타 생성 + 관절 좌표 저장
    ↓
움직임 에너지 / 관절 각도 / 3D 궤적 분석
    ↓
Streamlit 대시보드 출력
    """
)   