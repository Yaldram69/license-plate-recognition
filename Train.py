import os
import shutil
import yaml
import torch
from ultralytics import YOLO

# Define dataset paths
DATASET_PATH = r"../LicensePlate"
OUTPUT_DATASET_PATH = os.path.join(DATASET_PATH, "yolo_dataset")


def setup_yolo_dataset():
    """Prepares the dataset structure to be YOLO-compatible."""
    print("🔹 Starting dataset setup...")
    os.makedirs(OUTPUT_DATASET_PATH, exist_ok=True)

    # Define train, val, test folders for YOLO format
    for split in ['train', 'val', 'test']:
        img_src = os.path.join(DATASET_PATH, 'images', split)
        lbl_src = os.path.join(DATASET_PATH, 'labels', split)
        img_dest = os.path.join(OUTPUT_DATASET_PATH, split, 'images')
        lbl_dest = os.path.join(OUTPUT_DATASET_PATH, split, 'labels')

        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        print(f"📂 Created directories for {split}: Images -> {img_dest}, Labels -> {lbl_dest}")

        # Copy images and labels
        img_files = os.listdir(img_src)
        lbl_files = os.listdir(lbl_src)

        print(f"🔄 Copying {len(img_files)} images and {len(lbl_files)} label files for {split}...")
        for file in img_files:
            shutil.copy(os.path.join(img_src, file), os.path.join(img_dest, file))

        for file in lbl_files:
            shutil.copy(os.path.join(lbl_src, file), os.path.join(lbl_dest, file))
        print(f"✅ {split} data copied successfully!")

    # Create the YAML file required for YOLO
    data_yaml = {
        'train': os.path.join(OUTPUT_DATASET_PATH, 'train', 'images'),
        'val': os.path.join(OUTPUT_DATASET_PATH, 'val', 'images'),
        'test': os.path.join(OUTPUT_DATASET_PATH, 'test', 'images'),
        'nc': 1,  # Number of classes (0 = number plate)
        'names': ['license_plate']
    }

    yaml_path = os.path.join(OUTPUT_DATASET_PATH, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    print(f"📄 YAML file created at: {yaml_path}")

    print("✅ Dataset setup completed. Ready for training!")


def train_yolo_model(epochs=20, img_size=640, batch_size=16):
    """Trains YOLOv8 on the dataset with optimized memory usage."""
    print("\n🚀 Starting YOLOv8 training...")

    # Set device (GPU or CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔍 Using device: {device}")

    # Load YOLOv8 model (Nano version for lower VRAM usage)
    model = YOLO('yolov8n.pt')
    print(f"🛠 Training for {epochs} epochs with image size {img_size} and batch size {batch_size}...")

    # Start training
    model.train(
        data=os.path.join(OUTPUT_DATASET_PATH, 'data.yaml'),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=os.path.join(DATASET_PATH, "runs", "train"),
        name='license_plate_detection',
        device=device,
        amp=True  # Enable Automatic Mixed Precision (FP16)
    )

    print("🎯 Training completed! Model saved in 'runs/train/license_plate_detection'.")


if __name__ == "__main__":
    setup_yolo_dataset()
    train_yolo_model(epochs=20, img_size=640, batch_size=16)  # Lower batch size for stability
