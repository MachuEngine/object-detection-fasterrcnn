# object-detection-fasterrcnn
Faster R-CNN을 이용한 객체 검출


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
