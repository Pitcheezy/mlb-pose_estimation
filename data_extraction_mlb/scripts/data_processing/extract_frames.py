import cv2
import os
import glob

video_folder = '../data/raw/videos/ohtani_videos/**/*.mp4' # 모든 하위 폴더의 mp4 파일을 대상으로 함
output_folder = '../data/processed/dataset/images'
frame_interval = 10 # 10프레임마다 1장씩 저장 (숫자를 줄이면 더 많은 이미지 추출)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_files = glob.glob(video_folder, recursive=True)
print(f"총 {len(video_files)}개의 비디오 파일에서 이미지를 추출합니다.")

for video_path in video_files:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path).replace('.mp4', '')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            image_name = f"{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_folder, image_name), frame)
        
        frame_count += 1
    
    cap.release()
    print(f"'{video_path}' 처리 완료.")