import torchvision
from torchvision.datasets import VOCDetection
from torchvision import transforms

def get_dataset(data_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 추가 변환
    ])

    train_dataset = VOCDetection(root=data_path, year='2012', image_set='train', download=True, transform=transform)
    val_dataset = VOCDetection(root=data_path, year='2012', image_set='val', download=True, transform=transform)

    return train_dataset, val_dataset
