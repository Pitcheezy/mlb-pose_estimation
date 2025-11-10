import cv2
import numpy as np
from ultralytics import YOLO
import os

# --- 1. 사용할 모델 및 영상 경로 설정 ---

# 스크립트의 절대 경로를 기준으로 프로젝트 루트 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# ★★★ 수정 포인트 1 ★★★
# 성공적으로 훈련된 첫 번째 모델의 경로를 정확히 지정합니다.
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'pitcher_detector', 'runs', 'detect', 'train', 'weights', 'best.pt')

# ★★★ 수정 포인트 2 ★★★
# 분석하고 싶은 오타니 영상 파일들의 경로를 리스트로 지정합니다.
# ohtani_videos 폴더에서 여러 영상을 선택하여 분석합니다.
VIDEO_PATHS = [
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'videos', 'ohtani_videos', '2018', '2018-04-01_529450_atbat_13_pitch_1_ST_Sweeper_none.mp4'),
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'videos', 'ohtani_videos', '2018', '2018-04-01_529450_atbat_13_pitch_2_ST_Sweeper_none.mp4'),
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'videos', 'ohtani_videos', '2018', '2018-04-01_529450_atbat_13_pitch_3_ST_Sweeper_strikeout.mp4'),
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'videos', 'ohtani_videos', '2018', '2018-04-01_529450_atbat_14_pitch_1_FF_4-Seam_Fastball_none.mp4'),
    os.path.join(PROJECT_ROOT, 'data', 'raw', 'videos', 'ohtani_videos', '2018', '2018-04-01_529450_atbat_14_pitch_2_FF_4-Seam_Fastball_single.mp4')
]

# 분석 결과 저장 디렉토리
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'analyzed_videos')

# -------------------------------------------

# --- 필요한 도구들 초기화 ---
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"오류: YOLO 모델을 불러오는 데 실패했습니다. 경로를 확인하세요: {YOLO_MODEL_PATH}")
    print(f"상세 오류: {e}")
    exit()

# MediaPipe 대신 간단한 객체 탐지 기반 분석으로 변경

import os

# 분석 결과 저장 디렉토리 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📁 분석 결과 저장 디렉토리 생성: {OUTPUT_DIR}")

def analyze_single_video(video_path, yolo_model, output_dir):
    """단일 영상을 분석하여 결과를 파일로 저장합니다."""
    print(f"\n🎬 분석 시작: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 오류: 비디오 파일을 열 수 없습니다: {video_path}")
        return None

    # 비디오 저장 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 파일 경로 설정
    video_name = os.path.basename(video_path)
    output_path = os.path.join(output_dir, video_name.replace('.mp4', '_analyzed.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"📹 출력 파일: {output_path}")
    print(f"📊 비디오 정보: {fps}fps, {width}x{height}")

    # 프레임 카운터 및 탐지 통계
    frame_count = 0
    detection_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO로 투수 탐지
        results = yolo_model(frame, verbose=False)

        if results and results[0].boxes:
            # 모든 탐지된 객체에 대해 처리
            for box in results[0].boxes:
                if box.conf > 0.5:  # 신뢰도 50% 이상
                    detection_count += 1
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    # 탐지된 영역에 사각형 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 신뢰도 표시
                    confidence = box.conf.item() * 100
                    cv2.putText(frame, f"Pitcher: {confidence:.1f}%", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 프레임 정보 표시
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 처리된 프레임을 출력 비디오에 저장
        out.write(frame)

        # 진행 상황 출력 (50프레임마다)
        if frame_count % 50 == 0:
            print(f"⏳ 처리 중: {frame_count} 프레임 완료, 탐지: {detection_count}")

    cap.release()
    out.release()

    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0

    result = {
        'video_path': video_path,
        'output_path': output_path,
        'total_frames': frame_count,
        'detections': detection_count,
        'detection_rate': detection_rate
    }

    print(f"✅ 분석 완료: {os.path.basename(video_path)}")
    print(f"   📊 총 프레임: {frame_count}")
    print(f"   🎯 투수 탐지: {detection_count}")
    print(f"   📈 탐지율: {detection_rate:.1f}%")

    return result

# --- 여러 영상 배치 분석 시작 ---
print(f"\n🚀 총 {len(VIDEO_PATHS)}개의 영상 분석을 시작합니다!")
print("=" * 60)

all_results = []
successful_analyses = 0

for i, video_path in enumerate(VIDEO_PATHS, 1):
    print(f"\n[ {i}/{len(VIDEO_PATHS)} ] 번째 영상 처리 중...")
    print("-" * 50)

    # 각 영상 분석
    result = analyze_single_video(video_path, yolo_model, OUTPUT_DIR)

    if result:
        all_results.append(result)
        successful_analyses += 1
    else:
        print(f"❌ {video_path} 분석 실패")

# --- 최종 결과 요약 ---
print("\n" + "=" * 60)
print("🎉 모든 영상 분석 완료!")
print("=" * 60)
print(f"📊 분석한 영상 수: {len(VIDEO_PATHS)}개")
print(f"✅ 성공한 분석: {successful_analyses}개")
print(f"❌ 실패한 분석: {len(VIDEO_PATHS) - successful_analyses}개")

if all_results:
    total_frames = sum(r['total_frames'] for r in all_results)
    total_detections = sum(r['detections'] for r in all_results)
    avg_detection_rate = sum(r['detection_rate'] for r in all_results) / len(all_results)

    print("\n📈 전체 통계:")
    print(f"   총 프레임 수: {total_frames}")
    print(f"   총 탐지 수: {total_detections}")
    print(f"   평균 탐지율: {avg_detection_rate:.1f}%")

    print("\n📁 생성된 분석 영상들:")
    for result in all_results:
        video_name = os.path.basename(result['output_path'])
        print(f"   ✅ {video_name} (탐지율: {result['detection_rate']:.1f}%)")

print(f"\n💾 모든 결과는 '{OUTPUT_DIR}' 폴더에 저장되었습니다!")