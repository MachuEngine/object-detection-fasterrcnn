import os
import torch
import logging
from torch.utils.data import DataLoader

def train_model(model, train_dataset, val_dataset, epochs, learning_rate, batch_size, momentum, weight_decay, checkpoint_dir, save_interval):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    os.makedirs(checkpoint_dir, exist_ok=True)
    train_metrics = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        logging.info("Starting training epoch %d", epoch)

        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            
            # `targets` 전처리 - dictionary에서 tensor로 변환
            targets = [
                {
                    "boxes": torch.tensor(
                        [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                          int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])]
                         for obj in target['annotation']['object']],
                        dtype=torch.float32, device=device
                    ),
                    "labels": torch.tensor(
                        [1 for _ in target['annotation']['object']],  # 임시로 클래스 '1' 사용
                        dtype=torch.int64, device=device
                    ),
                    "image_id": torch.tensor([i], dtype=torch.int64, device=device)
                }
                for i, target in enumerate(targets)
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}")
        train_metrics.append(avg_loss)

    return train_metrics
