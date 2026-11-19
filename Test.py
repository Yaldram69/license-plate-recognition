import os
import cv2
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load PaddleOCR model with GPU
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=(device == "cuda"))

# Load trained YOLOv8 model on GPU
model_path = "./runs/train/license_plate_Final/weights/best.pt"
model = YOLO(model_path).to(device)

# Load video
video_path = "./a.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Video Writer to save output
output_path = "output_videoA.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on GPU
    results = model(frame, device=device)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = box.conf[0].item()  # Confidence score

            # Extract license plate region
            plate_img = frame[y1:y2, x1:x2]

            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

            # Run PaddleOCR (automatically uses GPU if available)
            detected_text = ""
            try:
                ocr_results = ocr.ocr(gray, cls=True)
                for result in ocr_results:
                    if result:
                        for line in result:
                            detected_text += line[1][0] + " "  # Extract recognized text
                detected_text = detected_text.strip()
            except Exception as e:
                print(f"⚠️ OCR Error: {e}")
                detected_text = "N/A"

            # Draw bounding box and recognized text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    # Save frames instead of showing
    cv2.imwrite(f"{frames_dir}/frame_{frame_count}.jpg", frame)  # Saves frames as images
    frame_count += 1



cap.release()
out.release()

print(f"✅ Output video saved as {output_path}")
print(f"✅ Frames saved in '{frames_dir}/' directory")
