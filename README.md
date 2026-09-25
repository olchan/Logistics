# CJ 미래기술챌린지 — Box Sizing (화물 크기 추정)

Synthetic 상차 CCTV 영상 내 화물 객체를 분석하여, 개별 화물의 가로(w)·세로(d)·높이(h)를 cm 단위로 추정하는 파이프라인입니다.

---

## 실행 방법

```bash
python main.py --input {video_folder_path}
```

- `{video_folder_path}` 아래의 모든 `.mp4` 파일을 읽어 추론을 수행합니다.
- 결과는 `main.py`와 같은 디렉토리에 `result.json`으로 저장됩니다.
- 모든 경로는 `os.path.dirname(os.path.abspath(__file__))` 기준 상대경로로 구성되어 있어, 입력 경로/파일명을 하드코딩하지 않습니다.

---

## 실행 환경

| 항목 | 내용 |
|---|---|
| 채점 GPU 환경 | NVIDIA A100 40G |
| ONNX Runtime | `onnxruntime-gpu==1.20.1`, `CUDAExecutionProvider` 우선, 실패 시 `CPUExecutionProvider`로 폴백 |
| 학습 환경 | Google Colab, NVIDIA A100 GPU (Box Detection·Rail Segmentation), NVIDIA L4 GPU (BoxRegressor) |
| 외부 네트워크 | 추론 시 사용하지 않음 (모든 모델은 로컬 ONNX 파일에서 로드) |
---

## 전체 파이프라인 개요

영상 한 편은 아래 순서로 처리됩니다.

1. **모션 기반 ROI 추출** — 프레임 간 배경 차분으로 컨베이어 벨트 활성 영역만 crop
2. **Box Detection** (YOLO26-Large, v5) — ROI 내 상자 검출
3. **2-Pass Tracking** (ByteTracker 기반) — 1차 트래킹으로 컨베이어 속도 추정 → 속도 기반 동적 파라미터로 2차 트래킹 → track 병합(부분 가림 재검출 처리 포함)
4. **Rail Segmentation** (YOLOv8s-seg) — 레일 영역을 사다리꼴(quad)로 추정, 픽셀→cm 스케일 산출
5. **Focal Length 추정** (f̂) — 레일 quad의 기하학적 특징으로부터 focal length를 선형회귀로 예측
6. **Depth Estimation** (Depth Anything V3, DA3METRIC-LARGE) — 각 트랙 crop에 대한 monocular depth 추정
7. **BoxRegressor** (ResNet18 기반, 5-fold 앙상블) — RGB+Depth 4채널 이미지와 (bbox scale, focal) 정보를 받아 (w,d,h) 비율 예측 → 실측 스케일을 곱해 최종 cm 단위 크기 산출

레일 검출이 실패한 경우, DA3 depth 기반 스케일(`SC_FALLBACK_K` 보정)로 대체합니다.

---

## 사용 모델 요약

- **YOLO26-Large v5** — Box Detection. 최종 추론에 사용됨. 학습 코드: `train_src/box_detector/`
- **YOLO26-Large v6** — BoxRegressor 학습 데이터(트래킹/크롭) 생성 전용, 최종 추론에는 사용되지 않음. 학습 코드: `train_src/box_detector/` (v5와 동일 코드, 데이터셋만 다름)
- **Grounding DINO** — Box 자동 라벨링 보조(Label Studio 수작업 전 단계), 최종 추론에는 사용되지 않음. 학습 코드: `train_src/box_detector/`
- **YOLOv8s-seg** — Rail Segmentation. 최종 추론에 사용됨. 학습 코드: `train_src/rail_detector/`
- **Depth Anything V3 (DA3METRIC-LARGE)** — Monocular Depth Estimation. 최종 추론에 사용됨. 별도 fine-tuning 없이 pretrained ONNX를 그대로 사용했으므로 `train_src/`에 해당 학습 코드가 없습니다.
- **ResNet18(ImageNet pretrained) → BoxRegressor** — (w,d,h) 비율 예측, 5-fold 앙상블. 최종 추론에 사용됨. 학습 코드: `train_src/box_regressor/`

---

## 학습 데이터 (수작업 라벨링 관련)

- Box Detection, Rail Segmentation 학습에 사용된 라벨은 대회 제공 Train 영상을 기반으로 **로컬 환경(Label Studio)에서 직접 수작업 레이블링**하여 생성했습니다.
- BoxRegressor 학습에 사용된 GT ↔ tracking 결과 매핑(`train_label_mapping.xlsx`)도 동일하게 로컬 환경에서 직접 작업했습니다.
- 위 원본 라벨링 데이터, 그리고 이로부터 파생된 중간 산출물(크롭 이미지, depth map 등 npz/pkl)들의 생성 로직은 각 하위 폴더의 노트북과 README에 전부 기술되어 있습니다.
- 대회가 제공한 데이터 외의 외부 데이터는 사용하지 않았습니다.

---

## checkpoints/ 구성

| 파일 | 역할 |
|---|---|
| `yolo26l_v5_augmentation.onnx` | Box Detection (`main.py`가 로드) |
| `rail_seg_yolov8s_LabelStudio_best.onnx` | Rail Segmentation |
| `DA3METRIC-LARGE.onnx` | Monocular Depth |
| `box_regressor_fold0~4.onnx` (+ 각 `.onnx.data`) | BoxRegressor 5-fold 앙상블 |
| `norm_stats.json` | Train 단계에서 결정된 정규화 상수 (추론 시 반드시 학습 때와 동일한 값을 사용해야 함) |

---

## train_src/ 구성
train_src/
├── box_detector/       ← YOLO26 Box Detection (v5: 최종 추론용 / v6: BoxRegressor 데이터 생성용)
├── rail_detector/      ← YOLOv8s-seg Rail Segmentation
└── box_regressor/      ← ResNet18 기반 BoxRegressor, 5-fold 앙상블 (v6 detector + rail/depth 모델을 활용해 학습 데이터 조립 후 학습)

각 폴더에는 학습 노트북(`.ipynb`), README, `pip freeze` 기반 `requirements_*.txt`가 포함되어 있습니다.

---

## 평가 관련 참고사항

- 입력 영상 매 프레임을 실제로 검출·트래킹·depth 추정하여 결과를 산출하며, 무작위 값이나 학습 데이터 통계값을 그대로 반환하는 방식은 사용하지 않았습니다.

---


## 전체 디렉토리 구조
​
```
demo.zip
│
├── main.py                                          ← 실행 진입점 
|
├── checkpoints/                                     ← inference에 실제 사용되는 ONNX 모델
│   ├── yolo26l_v5_augmentation.onnx                 ← Box Detection (main.py가 로드)
│   ├── rail_seg_yolov8s_LabelStudio_best.onnx        ← Rail Segmentation
│   ├── DA3METRIC-LARGE.onnx                          ← Monocular Depth
│   ├── box_regressor_fold0.onnx                      ← BoxRegressor 5-fold 앙상블
│   ├── box_regressor_fold0.onnx.data                 
│   ├── box_regressor_fold1.onnx
│   ├── box_regressor_fold1.onnx.data
│   ├── box_regressor_fold2.onnx
│   ├── box_regressor_fold2.onnx.data
│   ├── box_regressor_fold3.onnx
│   ├── box_regressor_fold3.onnx.data
│   ├── box_regressor_fold4.onnx
│   ├── box_regressor_fold4.onnx.data
│   └── norm_stats.json                               ← 정규화 상수 (Train 단계에서 결정됨)
│
├── train_src/                                        ← 모델 학습 코드 (모델별 폴더 분리)
│   │
│   ├── box_detector/
│   │   ├── train_box_detector.ipynb                  ← Grounding DINO를 통한 box 라벨링 → Label Studio에 업로드하여 수작업 수정 → YOLO26 fine-tuning
│   │   ├── README.md                                 ← Yolo 26 학습 파이프라인 설명, v5/v6 용도 구분, pretrained 출처 명시
│   │   └── requirements_box_detector.txt             ← pip freeze (ultralytics, transformers, torch 등 포함)
│   │
|   |
│   ├── rail_detector/
│   │   ├── train_rail_detector.ipynb                 ← Yolo v8 seg fine-tuning
│   │   ├── README.md                                 ← Yolo v8 seg 학습 파이프라인 설명
│   │   └── requirements_rail_detector.txt            ← pip freeze (ultralytics, transformers, torch 등 포함)
|   |
|   └── box_regressor/
│       ├── weights/
│       |   └── yolo26l_v6_manual_augmentation.onnx  ← 트래킹/크롭 생성에 쓰인 box detector : YOLO26 v6
│       |
|       ├── train_label_mapping.xlsx                 ← GT(w,d,h) ↔ YOLO26 v6 tracking 결과 매핑 엑셀
│       ├── train_box_regressor.ipynb                ← FINAL_BOXREGRESSOR_DATA.npz로 5-fold 학습
│       ├── README.md
│       └── requirements_training_box_regressor.txt  ← pip freeze (ultralytics, transformers, torch 등 포함)
│
└── README.md                                        ← 전체 파이프라인 개요, 사용 모델 요약
```