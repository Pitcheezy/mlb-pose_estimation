import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

def calculate_angle(a, b, c):
    """세 점 a, b, c가 주어졌을 때 점 b(팔꿈치)에서의 각도를 계산합니다."""
    # tensor를 numpy로 변환 (GPU tensor인 경우)
    if hasattr(a, 'cpu'):
        a = a.cpu().numpy()
    if hasattr(b, 'cpu'):
        b = b.cpu().numpy()
    if hasattr(c, 'cpu'):
        c = c.cpu().numpy()

    # numpy array로 변환
    a = np.asarray(a, dtype=np.float32) # 어깨
    b = np.asarray(b, dtype=np.float32) # 팔꿈치
    c = np.asarray(c, dtype=np.float32) # 손목

    # 벡터 계산
    ba = a - b  # 어깨에서 팔꿈치로의 벡터
    bc = c - b  # 손목에서 팔꿈치로의 벡터

    # 코사인 법칙을 사용한 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)  # 범위 제한
    angle = np.arccos(cosine_angle) * 180.0 / np.pi

    return angle

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
    pitcher_detector = YOLO(YOLO_MODEL_PATH)
    pose_estimator = YOLO('yolov8n-pose.pt')
except Exception as e:
    print(f"오류: YOLO 모델을 불러오는 데 실패했습니다. 경로를 확인하세요: {YOLO_MODEL_PATH}")
    print(f"상세 오류: {e}")
    exit()

# MediaPipe 대신 간단한 객체 탐지 기반 분석으로 변경

# 분석 결과 저장 디렉토리 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📁 분석 결과 저장 디렉토리 생성: {OUTPUT_DIR}")

def analyze_single_video(video_path, pitcher_detector, pose_estimator, output_dir):
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

    # --- ★ 1. 키(Key) 추출 및 분석 변수 초기화 ★ ---

    # 파일명에서 키(Key) 파싱 (예: 2018-04-01_529450_atbat_13_pitch_1_ST...)
    try:
        parts = os.path.basename(video_path).split('_')
        game_pk = int(parts[1])
        at_bat_number = int(parts[3])
        pitch_number = int(parts[5])
    except Exception as e:
        print(f"❌ 오류: 파일명에서 키를 파싱할 수 없습니다: {os.path.basename(video_path)} -> {e}")
        return None # 이 비디오 분석 중단

    frame_count = 0
    detection_count = 0

    # 릴리스 포인트 추적용 변수
    prev_wrist_pos = None
    max_wrist_velocity = -1
    angle_at_release = -1
    frame_at_release = -1

    analyzed_angles = [] # 프레임별 각도 저장 (평균 계산용)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        output_frame = frame.copy()

        # --- 1단계: 투수 탐지 ---

        detect_results = pitcher_detector(frame, verbose=False)

        if detect_results and detect_results[0].boxes:
            box = detect_results[0].boxes[0]

            if box.conf > 0.5:
                detection_count += 1 # (기존 코드에서 이동)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # --- 2단계: 자세 추정 (Crop) ---

                pad = 20
                crop_x1 = max(0, x1 - pad); crop_y1 = max(0, y1 - pad)
                crop_x2 = min(frame.shape[1], x2 + pad); crop_y2 = min(frame.shape[0], y2 + pad)
                pitcher_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if pitcher_crop.size == 0: continue

                pose_results = pose_estimator(pitcher_crop, verbose=False)
                annotated_crop = pose_results[0].plot() # 뼈대 그리기

                try:
                    if pose_results[0].keypoints and pose_results[0].keypoints.data.shape[1] == 17:
                        kpts = pose_results[0].keypoints.data[0] # (17, 3)

                        right_shoulder = kpts[6]
                        right_elbow = kpts[8]
                        right_wrist = kpts[10]

                        # 신뢰도 임계값 설정
                        confidence_threshold = 0.3  # 키포인트 검출 임계값

                        # 키포인트가 검출되었는지 확인
                        keypoints_detected = (right_shoulder[2] > confidence_threshold and
                                            right_elbow[2] > confidence_threshold and
                                            right_wrist[2] > confidence_threshold)

                        if keypoints_detected:
                            # (A) 각도 계산
                            try:
                                angle = calculate_angle(right_shoulder[:2], right_elbow[:2], right_wrist[:2])
                                analyzed_angles.append(angle) # 평균 계산용 저장

                                elbow_pos_crop = (int(right_elbow[0]), int(right_elbow[1]))
                                cv2.putText(annotated_crop, f"{angle:.1f}", (elbow_pos_crop[0] + 5, elbow_pos_crop[1]),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                            except Exception as e:
                                print(f"   [프레임 {frame_count}] 각도 계산 에러: {e}")
                                continue

                            # (B) 릴리스 포인트(최대 손목 속도) 계산
                            # tensor를 numpy로 변환
                            if hasattr(right_wrist[:2], 'cpu'):
                                current_wrist_pos = right_wrist[:2].cpu().numpy()
                            else:
                                current_wrist_pos = np.asarray(right_wrist[:2])

                            # 릴리스 포인트 추적을 위한 초기화
                            if prev_wrist_pos is None:
                                prev_wrist_pos = current_wrist_pos
                                continue

                            # 유클리드 거리로 속도 근사 (픽셀 단위)
                            velocity = np.linalg.norm(current_wrist_pos - prev_wrist_pos)

                            # 최대 속도 업데이트 및 릴리스 포인트 감지
                            if velocity > max_wrist_velocity:
                                max_wrist_velocity = velocity
                                angle_at_release = angle
                                frame_at_release = frame_count

                            prev_wrist_pos = current_wrist_pos

                    # 뼈대와 각도가 그려진 crop을 원본 프레임에 다시 붙여넣기
                    output_frame[crop_y1:crop_y2, crop_x1:crop_x2] = annotated_crop

                except Exception as e:
                    pass

        # (시각화) 릴리스 프레임 표시
        if frame_count == frame_at_release:
            cv2.putText(output_frame, "RELEASE!", (50, 80), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

        cv2.putText(output_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(output_frame)

    # --- ★ 2. 루프 종료 후 결과 반환 ★ ---

    cap.release()
    out.release()

    avg_angle = np.mean(analyzed_angles) if analyzed_angles else -1

    # 최종 분석 결과를 딕셔너리로 반환
    result_data = {
        'game_pk': game_pk,
        'at_bat_number': at_bat_number,
        'pitch_number': pitch_number,
        'calculated_release_angle': angle_at_release,
        'calculated_avg_angle': avg_angle,
        'release_frame': frame_at_release,
        'max_wrist_velocity': max_wrist_velocity,
        'output_video_path': output_path,
        'detection_rate': (detection_count / frame_count) * 100 if frame_count > 0 else 0
    }

    print(f"✅ 분석 완료: {os.path.basename(video_path)}")
    print(f"   🔑 Keys: {game_pk}, {at_bat_number}, {pitch_number}")
    print(f"   🚀 릴리스 각도: {angle_at_release:.2f} (at frame {frame_at_release})")
    print(f"   📊 평균 각도: {avg_angle:.2f}")

    return result_data # 기존 result 딕셔너리 대신 이 딕셔너리를 반환

# --- 여러 영상 배치 분석 시작 ---
print(f"\n🚀 총 {len(VIDEO_PATHS)}개의 영상 분석을 시작합니다!")
print("=" * 60)

all_results = []
successful_analyses = 0

for i, video_path in enumerate(VIDEO_PATHS, 1):
    print(f"\n[ {i}/{len(VIDEO_PATHS)} ] 번째 영상 처리 중...")
    print("-" * 50)

    # 각 영상 분석
    result = analyze_single_video(video_path, pitcher_detector, pose_estimator, OUTPUT_DIR)

    if result:
        all_results.append(result)
        successful_analyses += 1
    else:
        print(f"❌ {video_path} 분석 실패")

# --- 최종 결과 요약 및 CSV 저장 ---

print("\n" + "=" * 60)
print("🎉 모든 영상 분석 완료!")
print("=" * 60)

successful_analyses = len(all_results)
print(f"📊 분석한 영상 수: {len(VIDEO_PATHS)}개")
print(f"✅ 성공한 분석: {successful_analyses}개")
print(f"❌ 실패한 분석: {len(VIDEO_PATHS) - successful_analyses}개")

if all_results:
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results)

    # CSV 파일로 저장
    csv_output_path = os.path.join(PROJECT_ROOT, 'results', 'video_analysis_results.csv')
    results_df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')

    print("\n📈 전체 통계:")
    print(f"   평균 릴리스 각도: {results_df['calculated_release_angle'].mean():.2f}")
    print(f"   평균 탐지율: {results_df['detection_rate'].mean():.1f}%")

    print(f"\n💾 ★★★ 분석 결과가 CSV 파일로 저장되었습니다! ★★★")
    print(f"   {csv_output_path}")
else:
    print("\n분석에 성공한 데이터가 없습니다.")