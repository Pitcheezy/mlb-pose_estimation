# Shohei Ohtani MLB Pitch Analysis Project

이 프로젝트는 Shohei Ohtani의 투구 동작을 분석하기 위한 머신러닝 파이프라인입니다. YOLOv8을 사용한 투수 감지와 CNN을 사용한 구종 분류를 포함합니다.

## 📁 프로젝트 구조

```
pitcheezy_mlb_pitch_analysis/
├── scripts/                          # Python 스크립트들
│   ├── data_collection/              # 데이터 수집 스크립트
│   │   ├── get_ohtani_data.py        # MLB 데이터 수집
│   │   ├── mlb_data_fetcher.py       # 경기 데이터 추출
│   │   └── download_ohtani_videos.py # 영상 다운로드
│   ├── data_processing/              # 데이터 전처리 스크립트
│   │   ├── extract_frames.py         # 비디오 프레임 추출
│   │   └── sample_images.py          # 이미지 샘플링
│   ├── model_training/               # 모델 훈련 스크립트
│   │   ├── pitch_type_classifier.py  # 구종 분류 CNN 모델
│   │   └── train_pitcher_detector.py # 투수 감지 YOLO 모델
│   ├── analysis/                     # 분석 스크립트
│   │   └── analyze_video.py          # 비디오 분석 및 결과 생성
│   └── utils/                        # 유틸리티 스크립트
│       ├── check_roboflow_workspace.py
│       ├── test_roboflow_api.py
│       └── upload_to_roboflow.py
├── data/                             # 데이터 파일들
│   ├── raw/                          # 원본 데이터
│   │   ├── csv/                      # MLB 통계 데이터
│   │   └── videos/                   # 다운로드된 영상
│   ├── processed/                    # 전처리된 데이터
│   │   ├── pitch_type_dataset.csv    # 구종 분류용 데이터셋
│   │   └── dataset/                  # 이미지 데이터셋
│   └── roboflow/                     # Roboflow 데이터셋
├── models/                           # 훈련된 모델들
│   ├── pitch_classifier/             # 구종 분류 모델
│   │   ├── best_pitch_classifier.h5
│   │   ├── confusion_matrix.png
│   │   └── training_history.png
│   ├── pitcher_detector/             # 투수 감지 모델
│   │   ├── runs/                     # YOLO 훈련 결과들
│   │   └── yolo11n.pt
│   └── pretrained/                   # 사전 학습 모델
│       └── yolov8n.pt
├── results/                          # 분석 결과
│   └── analyzed_videos/              # 분석된 영상 파일들
└── README.md                         # 프로젝트 설명
```

## 🚀 사용 방법

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# pip install -r requirements.txt  # 필요한 패키지 설치
```

### 2. 데이터 수집
```bash
cd scripts/data_collection
python get_ohtani_data.py              # MLB 데이터 수집
python download_ohtani_videos.py       # 영상 다운로드
```

### 3. 데이터 전처리
```bash
cd ../data_processing
python extract_frames.py               # 프레임 추출
python sample_images.py                # 이미지 샘플링
```

### 4. 모델 훈련
```bash
cd ../model_training
python train_pitcher_detector.py       # YOLO 투수 감지 모델
python pitch_type_classifier.py        # CNN 구종 분류 모델
```

### 5. 영상 분석
```bash
cd ../analysis
python analyze_video.py                # 비디오 분석
```

## 📊 현재 성능

### YOLOv8 투수 감지 모델
- **평균 탐지율**: 83.1%
- **총 분석 프레임**: 3,240개
- **총 탐지 수**: 2,454회

### CNN 구종 분류 모델
- **테스트 정확도**: 43%
- **주요 구종**: FF (4-Seam Fastball), ST (Sweeper), CU (Curveball)
- **데이터셋 크기**: 110,000+ 샘플

## 🛠️ 기술 스택

- **프로그래밍 언어**: Python 3.13
- **머신러닝 프레임워크**: TensorFlow/Keras, PyTorch
- **컴퓨터 비전**: OpenCV, Ultralytics YOLOv8
- **데이터 처리**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **GPU**: NVIDIA CUDA 12.1 (RTX 4070)

## 📈 향후 개선 방향

1. **모델 성능 향상**
   - 더 큰 데이터셋 수집
   - 모델 앙상블 기법 적용
   - 하이퍼파라미터 최적화

2. **실시간 분석**
   - 스트리밍 영상 분석
   - 실시간 구종 예측

3. **추가 기능**
   - 투구 궤적 분석
   - 속도 및 회전수 예측
   - 상대 투수 비교 분석

## 📄 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 사용됩니다.
