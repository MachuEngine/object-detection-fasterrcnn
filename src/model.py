import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes, pretrained=True):
    """
    * Faster R-CNN 모델을 생성하는 함수
    Faster R-CNN은 Faster라는 이름처럼 이전 모델(R-CNN, Fast R-CNN)의 속도 문제를 해결하며, 실시간 객체 탐지에 가까운 성능을 제공합니다.
    
    * 구성 요소
    1. Feature Extraction (특징 추출):
        - 백본 네트워크(예: ResNet, VGG)를 사용하여 입력 이미지에서 **특징 맵(feature map)**을 추출합니다.
    2. Region Proposal Network (RPN):
        - RPN은 입력된 특징 맵에서 **객체가 있을 가능성이 높은 영역(Region Proposal)**을 예측합니다.
        - Selective Search(이전 모델에서 사용된 느린 알고리즘)를 대체하여 속도를 크게 개선했습니다.
    3. ROI Head:
        - Region Proposal을 기반으로 객체의 클래스와 **경계 상자(Bounding Box)**를 예측합니다.
        - ROI Pooling 또는 ROI Align을 통해 Region Proposal을 고정된 크기로 변환하여 분류 및 위치 회귀를 수행합니다.
    
    * 동작 과정
    1. 입력 이미지:
        - 입력 이미지(예: 800x600)를 백본 네트워크(예: ResNet-50)에 전달
    2. 특징 맵 생성:
        - 백본 네트워크가 입력 이미지를 처리하여 고차원 특징 맵(예: 256x50x37)을 생성
    3. Region Proposal Network (RPN):
        - RPN이 특징 맵의 각 위치에서 **앵커 박스(Anchor Box)**를 생성
        - 각 앵커 박스에 대해:
            - 객체성 점수(Objectness Score): 객체인지 아닌지 예측.
            - 위치 조정(Regression): 앵커 박스의 위치를 조정.
    4. Non-Maximum Suppression (NMS):
        - RPN에서 생성된 여러 Region Proposal 중 겹치는 영역을 제거하고 상위 N개의 후보를 선택.
    5. ROI Pooling:
        - 선택된 Region Proposal을 고정 크기(예: 7x7)로 변환.
    6. ROI Head:
        - 각 ROI에서:
            - 객체의 클래스 분류.
            - 경계 상자 좌표(Bounding Box) 조정.

    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
