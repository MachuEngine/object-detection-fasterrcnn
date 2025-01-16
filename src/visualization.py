import matplotlib.pyplot as plt

def visualize_data(dataset, num_images=5):
    for i in range(num_images):
        image, target = dataset[i]
        plt.figure(figsize=(8, 6))
        img = image.permute(1, 2, 0).numpy() if hasattr(image, 'permute') else image.numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
        plt.show()

def visualize_results(train_metrics, eval_metrics):
    epochs = range(1, len(train_metrics) + 1)
    plt.figure()
    plt.plot(epochs, train_metrics, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    print("Evaluation Metrics:")
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value}")
