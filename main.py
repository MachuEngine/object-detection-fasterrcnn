import argparse
import logging
import os

from src.utils import load_config, setup_logging
from src.dataset import get_dataset
from src.model import get_model
from src.train import train_model
from src.eval import evaluate_model
from src.visualization import visualize_data, visualize_results

def main(config):
    # 로깅 설정
    setup_logging(config['logging']['file'], level=getattr(logging, config['logging']['level']))

    logging.info("Starting Faster R-CNN training pipeline")

    # 데이터셋 로드
    train_dataset, val_dataset = get_dataset(config['data']['path'])

    # 모델 생성
    model = get_model(num_classes=config['data']['num_classes'], pretrained=config['model']['pretrained'])

    # 데이터 시각화 (학습 전 샘플)
    visualize_data(train_dataset)

    # 모델 학습
    train_metrics = train_model(
        model, 
        train_dataset, 
        val_dataset, 
        epochs=config['train']['epochs'], 
        learning_rate=config['train']['learning_rate'],
        batch_size=config['train']['batch_size'],
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay'],
        checkpoint_dir=config['checkpoint']['directory'],
        save_interval=config['checkpoint']['save_interval']
    )

    # 모델 평가
    eval_metrics = evaluate_model(model, val_dataset)

    # 학습 및 평가 결과 시각화
    visualize_results(train_metrics, eval_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster R-CNN Training Script")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="구성 파일 경로")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
