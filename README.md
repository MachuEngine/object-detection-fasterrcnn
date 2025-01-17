# Faster R-CNN
Faster R-CNN을 이용한 객체 검출 (paper : https://arxiv.org/abs/1506.01497)


## 프로젝트 구조
```bash
project/
├── configs/
│   └── config.yaml
├── data/                # 데이터셋 폴더
├── logs/                # 로그 파일 폴더
├── checkpoints/         # 모델 체크포인트 저장 폴더
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── eval.py
│   ├── visualization.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## 설치
### Git 
```bash
https://github.com/MachuEngine/object-detection-fasterrcnn.git
```
### Requirements
```bash
pip install -r requirements.txt
```

## 학습

### 하이퍼 파라미터 
```bash
data:
  path: "./data"
  num_classes: 2

train:
  epochs: 10
  learning_rate: 0.005
  batch_size: 2
  momentum: 0.9
  weight_decay: 0.0005

model:
  pretrained: true

logging:
  level: INFO
  file: "./logs/train.log"

checkpoint:
  directory: "./checkpoints"
  save_interval: 5
```

### 실행
```bash
python main.py --config configs/config.yaml
```

---
### Faster R-CNN vs 이전 모델

| **특징**            | **R-CNN**                       | **Fast R-CNN**                     | **Faster R-CNN**                     |
|---------------------|---------------------------------|------------------------------------|---------------------------------------|
| **Region Proposal** | Selective Search               | Selective Search                  | **Region Proposal Network (RPN)**   |
| **속도**            | 매우 느림                      | 빠름                              | 더 빠름                               |
| **네트워크 구조**    | 분리된 CNN, SVM, 회귀          | 단일 네트워크                     | 단일 네트워크                         |
| **ROI 처리 방식**    | CNN에 개별 전달                | ROI Pooling                       | ROI Pooling                          |


### Faster R-CNN의 동작 과정
# Faster R-CNN 동작 과정

Faster R-CNN은 입력 이미지에서 객체를 탐지하고 최종적으로 원본 이미지에 결과를 매핑하기까지 여러 단계를 거칩니다. 아래는 각 단계의 역할과 출력 크기를 포함한 상세한 동작 과정입니다.

---

#### **1. 입력 이미지 처리**
- **입력**: 원본 이미지 (\(H_{\text{img}} \times W_{\text{img}} \times 3\), 예: \(800 \times 600 \times 3\)).
- **과정**:
  1. 이미지 크기 조정(Resizing): 가장 긴 변을 800 픽셀로 조정.
  2. 정규화(Normalization): 픽셀 값을 \(0-1\) 범위로 변환하거나 평균값/표준편차로 정규화.
- **출력**: 전처리된 이미지 (\(800 \times 600 \times 3\)).

---

#### **2. 백본 네트워크를 통한 특징 추출**
- **입력**: 전처리된 이미지 (\(800 \times 600 \times 3\)).
- **과정**:
  1. ResNet, VGG 등 백본 네트워크를 사용하여 **특징 맵** 생성.
  2. 합성곱과 풀링 과정을 통해 이미지의 고수준 정보를 추출.
- **출력 (특징 맵)**:
  - 크기: \(C \times H_{\text{feat}} \times W_{\text{feat}}\)
    - 예: \(256 \times 50 \times 37\) (stride=16 기준).

---

#### **3. Region Proposal Network (RPN)**
- **입력**: 특징 맵 (\(256 \times 50 \times 37\)).
- **과정**:
  1. 각 위치에서 9개의 **앵커 박스** 생성.
     - 총 앵커 박스 개수: \(H_{\text{feat}} \times W_{\text{feat}} \times 9\) (예: \(50 \times 37 \times 9 = 16,650\)).
  2. 각 앵커 박스에 대해:
     - **객체성 점수(Objectness Score)**: 객체인지 배경인지 예측.
     - **경계 상자 조정 값(Regression Offset)**: 앵커 박스를 객체에 맞게 조정.
  3. Non-Maximum Suppression(NMS):
     - 겹치는 박스를 제거하고 상위 \(N\)개의 Region Proposal 선택 (예: \(N=300\)).
- **출력**:
  - 상위 300개의 Region Proposal (\(300 \times 4\)).

---

#### **4. ROI Pooling**
- **입력**:
  - 특징 맵 (\(256 \times 50 \times 37\)).
  - Region Proposal (\(300 \times 4\)).
- **과정**:
  1. Region Proposal을 특징 맵에 매핑.
  2. 각 Proposal을 고정 크기(예: \(7 \times 7\))로 변환.
- **출력**:
  - 고정 크기 특징 맵 (\(300 \times 256 \times 7 \times 7\)).

---

#### **5. ROI Head (분류 및 경계 상자 조정)**
- **입력**: ROI Pooling 출력 (\(300 \times 256 \times 7 \times 7\)).
- **과정**:
  1. Flatten: \(7 \times 7\) 텐서를 1D 벡터로 변환 (\(300 \times 256 \times 49\)).
  2. 완전 연결 계층(FC Layers):
     - **클래스 분류(Classification)**: 각 Region Proposal의 객체 클래스와 확률 예측.
     - **경계 상자 조정(Regression)**: 각 Proposal의 위치를 조정.
- **출력**:
  - 클래스: \(300 \times K\) (\(K\): 클래스 개수, 예: 21).
  - 경계 상자: \(300 \times 4K\) (클래스별 4개의 좌표).

---

#### **6. 최종 경계 상자를 원본 이미지로 매핑**
- **입력**:
  - ROI Head 출력 (특징 맵 좌표 기준 경계 상자).
  - 다운샘플링 비율 (stride=16).
- **과정**:
  1. 특징 맵 좌표를 원본 이미지 좌표로 변환:
     - \(x_{\text{orig}} = x_{\text{feat}} \times \text{stride}\).
     - \(y_{\text{orig}} = y_{\text{feat}} \times \text{stride}\).
     - \(w_{\text{orig}} = w_{\text{feat}} \times \text{stride}\).
     - \(h_{\text{orig}} = h_{\text{feat}} \times \text{stride}\).
  2. 경계 상자 회귀 결과를 적용하여 위치와 크기 보정:
     - \(x = x_a + \Delta x \times w_a\),
     - \(y = y_a + \Delta y \times h_a\),
     - \(w = w_a \times e^{\Delta w}\),
     - \(h = h_a \times e^{\Delta h}\).
- **출력**:
  - 원본 이미지 좌표로 변환된 경계 상자.

---

#### **최종 출력**
- **결과**:
  - 객체 클래스(예: 사람, 자동차).
  - 원본 이미지에서의 경계 상자 좌표 (예: \(x, y, w, h\)).

---

#### **단계별 크기 요약**

| **단계**                    | **입력 크기**              | **출력 크기**                   |
|-----------------------------|--------------------------|---------------------------------|
| 원본 이미지                | \(800 \times 600 \times 3\) | -                               |
| 백본 특징 맵 생성          | \(800 \times 600 \times 3\) | \(256 \times 50 \times 37\)     |
| RPN 앵커 박스 생성          | \(256 \times 50 \times 37\) | \(300 \times 4\) (Region Proposal) |
| ROI Pooling                | \(256 \times 50 \times 37\) + \(300 \times 4\) | \(300 \times 256 \times 7 \times 7\) |
| ROI Head (분류 및 회귀)      | \(300 \times 256 \times 7 \times 7\) | 클래스: \(300 \times K\), 경계 상자: \(300 \times 4K\) |

---

#### **요약**
- Faster R-CNN은 **CNN 특징 추출 → RPN(Region Proposal) → ROI Pooling → 클래스 분류 및 경계 상자 조정** 단계를 통해 객체를 탐지합니다.
- 최종적으로 **특징 맵 좌표를 원본 이미지 좌표로 매핑**하여 객체의 위치를 정확히 표시합니다.

