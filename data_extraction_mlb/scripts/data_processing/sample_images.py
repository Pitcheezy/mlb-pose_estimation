import os
import random
import shutil
from tqdm import tqdm

source_folder = '../data/processed/dataset/images'
destination_folder = '../data/processed/dataset/for_labeling'
num_samples = 1000  # 라벨링할 이미지 개수 (필요에 따라 조절)

# 대상 폴더가 없으면 생성
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 소스 폴더의 모든 이미지 파일 목록 가져오기
try:
    all_images = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if len(all_images) < num_samples:
        print(f"오류: 원본 이미지 개수({len(all_images)})가 샘플링할 개수({num_samples})보다 적습니다.")
        exit()
    
    print(f"총 {len(all_images)}개의 이미지에서 {num_samples}개를 무작위로 샘플링합니다...")

    # 무작위로 이미지 선택
    sampled_images = random.sample(all_images, num_samples)

    # 선택된 이미지를 새 폴더로 복사
    for image_name in tqdm(sampled_images, desc="이미지 복사 중"):
        source_path = os.path.join(source_folder, image_name)
        destination_path = os.path.join(destination_folder, image_name)
        shutil.copy(source_path, destination_path)
        
    print(f"\n성공! '{destination_folder}' 폴더에 {num_samples}개의 이미지를 복사했습니다.")
    print("이제 이 폴더를 Roboflow에 업로드하세요.")

except FileNotFoundError:
    print(f"오류: '{source_folder}'를 찾을 수 없습니다. 프레임 추출이 올바르게 완료되었는지 확인하세요.")
except Exception as e:
    print(f"알 수 없는 오류가 발생했습니다: {e}")