import torch

def evaluate_model(model, val_dataset):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    eval_metrics = {}
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            # outputs와 targets를 비교해 평가 지표 계산
            break

    eval_metrics['example_metric'] = 0.0
    return eval_metrics
