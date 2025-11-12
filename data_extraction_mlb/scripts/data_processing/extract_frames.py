import cv2
import os
import glob
import numpy as np

video_folder = '../data/raw/videos/ohtani_videos/**/*.mp4' # 모든 하위 폴더의 mp4 파일을 대상으로 함
output_folder = '../data/processed/dataset/images'
frame_interval = 10 # 10프레임마다 1장씩 저장 (숫자를 줄이면 더 많은 이미지 추출)

# 블러처리 설정
BLUR_FACES = True  # 얼굴 블러처리 활성화/비활성화
BLUR_STRENGTH = 15  # 블러 강도 (홀수여야 함)

# 얼굴 검출을 위한 Haar Cascade 분류기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def blur_faces(frame):
    """
    프레임에서 얼굴을 검출하여 블러처리합니다.
    야구 영상에서는 얼굴이 작을 수 있으므로 여러 스케일로 검출합니다.
    """
    if not BLUR_FACES:
        return frame

    # 그레이스케일 변환 (얼굴 검출용)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출 (여러 스케일로 검출하여 작은 얼굴도 찾기)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(20, 20),  # 야구 영상에서는 얼굴이 작을 수 있음
        maxSize=(200, 200)  # 너무 큰 영역은 제외
    )

    # 각 얼굴에 블러 적용
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출
        face_roi = frame[y:y+h, x:x+w]

        # 블러 적용
        blurred_face = cv2.GaussianBlur(face_roi, (BLUR_STRENGTH, BLUR_STRENGTH), 0)

        # 블러된 얼굴을 원본 프레임에 적용
        frame[y:y+h, x:x+w] = blurred_face

    return frame

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_files = glob.glob(video_folder, recursive=True)
print(f"총 {len(video_files)}개의 비디오 파일에서 이미지를 추출합니다.")

if BLUR_FACES:
    print("🔍 얼굴 블러처리가 활성화되었습니다.")
    print(f"   블러 강도: {BLUR_STRENGTH}")
else:
    print("ℹ️  얼굴 블러처리가 비활성화되었습니다.")

total_processed = 0
for i, video_path in enumerate(video_files):
    print(f"\n📹 [{i+1}/{len(video_files)}] '{os.path.basename(video_path)}' 처리 중...")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path).replace('.mp4', '')
    video_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # 블러처리 적용 (활성화된 경우)
            processed_frame = blur_faces(frame.copy())

            image_name = f"{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_folder, image_name), processed_frame)
            video_processed += 1

        frame_count += 1

    cap.release()
    total_processed += video_processed
    print(f"   ✅ {video_processed}개 이미지 저장 완료")

print("
🎉 전체 처리 완료!")
print(f"   총 저장된 이미지: {total_processed}개")
print(f"   출력 폴더: {os.path.abspath(output_folder)}")