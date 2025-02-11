
import cv2
import os

# Video file path
video_path = r'C:\Users\kavit\OneDrive\Desktop\BOLT AI\dataset\fighting\VID-20241225-WA0004.mp4'
output_frames_dir = 'C:\Users\kavit\OneDrive\Desktop\chinmayee1\BOLT AI (3)\BOLT AI\backend\processedvideos'
os.makedirs(output_frames_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_interval = max(frame_count // 30, 1)  # Calculate interval for 30 evenly spaced frames

selected_frames = []
current_frame = 0

for i in range(30):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(output_frames_dir, f'frame_{i:02d}.jpg')
    cv2.imwrite(frame_path, frame)
    selected_frames.append(frame_path)
    current_frame += frame_interval

cap.release()
print(f'Extracted {len(selected_frames)} frames.')



from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Load model and processor
model = ViTForImageClassification.from_pretrained('csr2000/UCF_Crime')
processor = ViTImageProcessor(do_resize=True,
                              size=224,
                              do_normalize=True,
                              image_mean=[0.5, 0.5, 0.5],
                              image_std=[0.5, 0.5, 0.5])

# List of extracted frames
import os
output_frames_dir = r'C:\Users\kavit\OneDrive\Desktop\chinmayee1\BOLT AI (3)\BOLT AI\backend\frames'
selected_frames = sorted(os.listdir(output_frames_dir))

# User selects the frame number
frame_number = int(input(f"Enter the frame number (0-{len(selected_frames)-1}): "))

# Validate frame number
if frame_number < 0 or frame_number >= len(selected_frames):
    raise ValueError("Invalid frame number. Please choose a valid number from the range.")

# Select the specific frame
selected_frame_path = os.path.join(output_frames_dir, selected_frames[frame_number])
print(f'Selected Frame: {selected_frame_path}')

# Process the selected frame
image = Image.open(selected_frame_path).convert('RGB')
inputs = processor(images=image, return_tensors='pt')

# Perform inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Get prediction
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
predicted_class_label = model.config.id2label[str(predicted_class_idx)]

print(f'Predicted Class: {predicted_class_label}')