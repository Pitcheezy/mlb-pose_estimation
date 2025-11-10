# --------------------------------------------------
# 1. Roboflow에서 복사한 코드를 그대로 붙여넣기
# --------------------------------------------------
# pip install roboflow # 이 줄은 주석 처리하거나 지워도 됩니다.

from roboflow import Roboflow
# api_key="YOUR_API_KEY" 부분에 실제 발급받은 키를 넣어야 합니다.
rf = Roboflow(api_key="Bubuom4F7MItqMMTXSxz")
project = rf.workspace("pitcheezy").project("pitcher-pointing-wcoix")
version = project.version(1)
dataset = version.download("yolov8") # 데이터셋 다운로드 실행

# --------------------------------------------------
# 2. 기존 YOLO 훈련 코드 수정하기
# --------------------------------------------------
from ultralytics import YOLO

# YOLOv8n (nano) 모델을 기반으로 시작합니다.
model = YOLO('yolov8n.pt')

# ★★★ 가장 중요한 수정 ★★★
# data 인자 값을 Roboflow에서 다운로드한 데이터셋의 data.yaml 경로로 자동 설정합니다.
# dataset.location은 다운로드된 데이터셋 폴더의 경로를 의미합니다.
yaml_path = f"{dataset.location}/data.yaml"

# 수정된 경로를 사용하여 훈련을 시작합니다.
# data 인자를 자동으로 설정된 yaml_path로 지정하고, CUDA GPU를 사용하도록 설정합니다.
results = model.train(data=yaml_path, epochs=50, imgsz=640, device='cuda')

print("훈련이 완료되었습니다. 결과는 'runs/detect/train' 폴더에 저장됩니다.")