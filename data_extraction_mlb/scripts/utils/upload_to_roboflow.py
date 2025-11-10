import os
import glob
from roboflow import Roboflow
from tqdm import tqdm

# Roboflow 설정
API_KEY = "Bubuom4F7MItqMMTXSxz"  # Roboflow 계정에서 API 키를 가져와야 합니다
WORKSPACE = "pitcheezy"  # 소문자로 수정
PROJECT = "pitcher-pointing-wcoix"

# 업로드할 폴더
image_folder = "dataset/for_labeling"

def upload_images_to_roboflow():
    try:
        # Roboflow 초기화
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)

        # 이미지 파일 목록 가져오기
        image_files = glob.glob(os.path.join(image_folder, "*.jpg"))
        image_files.extend(glob.glob(os.path.join(image_folder, "*.jpeg")))
        image_files.extend(glob.glob(os.path.join(image_folder, "*.png")))

        if not image_files:
            print(f"ERROR: No image files found in '{image_folder}' folder.")
            return

        print(f"Uploading {len(image_files)} images to Roboflow...")
        print(f"Target project: {WORKSPACE}/{PROJECT}")

        # 각 이미지 업로드
        uploaded_count = 0
        for image_path in tqdm(image_files, desc="Uploading"):
            try:
                # 이미지 업로드 (라벨링용으로 train 세트에 추가)
                project.upload(image_path, batch_name="ohtani_pitch_frames")
                uploaded_count += 1

            except Exception as e:
                print(f"Upload failed: {os.path.basename(image_path)} - {e}")
                continue

        print(f"\nUpload completed: {uploaded_count}/{len(image_files)} successful")

    except Exception as e:
        print(f"Roboflow connection error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key at: https://app.roboflow.com/settings/api")
        print("2. Verify API_KEY variable has correct key")
        print("3. Confirm workspace and project names are correct")

if __name__ == "__main__":
    print("Roboflow Image Upload Script")
    print("=" * 50)

    # API 키가 설정되지 않은 경우 안내
    if API_KEY == "YOUR_API_KEY":
        print("ERROR: API key is not configured!")
        print("\nSetup instructions:")
        print("1. Go to https://app.roboflow.com/settings/api to get your API key")
        print("2. Replace API_KEY variable with your actual key")
        print("3. Run: python upload_to_roboflow.py")
        exit()

    upload_images_to_roboflow()
