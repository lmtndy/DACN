from ultralytics import YOLO
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # Verify CUDA availability
    print(torch.cuda.is_available())  # Should return True
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))  # Should display your GPU name

    # Check if GPU is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # Ensure the correct model file exists

    # Move model to GPU
    model.to(device)

    # Train the model
    data_yaml_path = './data.yaml'  # Ensure the correct path to the dataset YAML
    history = model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=1000,
        batch=16,
        save_period=1,  # Save after every epoch
        project='yolo_project',  # Valid project name without special characters
        name='yolo_train',  # Subfolder name
    )

    # Path to results.csv
    results_path = './yolo_project/yolo_train/results.csv'

    # Load the results.csv
    df = pd.read_csv(results_path)

    # Plotting Losses
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['box_loss'], label='Box Loss', marker='o')
    plt.plot(df['epoch'], df['cls_loss'], label='Class Loss', marker='o')
    plt.plot(df['epoch'], df['dfl_loss'], label='DFL Loss', marker='o')  # DFL = Distribution Focal Loss
    plt.title('Loss Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting Precision, Recall, and F1-Score
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['precision'], label='Precision', marker='o')
    plt.plot(df['epoch'], df['recall'], label='Recall', marker='o')
    plt.plot(df['epoch'], (2 * df['precision'] * df['recall']) / (df['precision'] + df['recall']),
             label='F1-Score', marker='o')
    plt.title('Precision, Recall, and F1-Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid()
    plt.show()

    # Plotting mAP (Mean Average Precision)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['mAP_50'], label='mAP@50', marker='o')
    plt.plot(df['epoch'], df['mAP_50-95'], label='mAP@50-95', marker='o')
    plt.title('mAP Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid()
    plt.show()

    # Save the final metrics plot
    plt.savefig('loss_metrics.png')
