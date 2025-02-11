import cv2
import os

def preprocess_video_to_frames(video_path, num_frames=50):
    # Create the output folder if it doesn't exist
    output_folder = r'C:\Users\kavit\OneDrive\Desktop\chinmayee1\BOLT AI (3)\BOLT AI\backend\processedvideos' 
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the frame interval to evenly select num_frames
    frame_interval = max(total_frames // num_frames, 1)
    
    frame_count = 0
    extracted_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame if it's one of the evenly spaced ones
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_frames += 1
            
            # Stop if we have reached the desired number of frames
            if extracted_frames >= num_frames:
                break
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_frames} frames to {output_folder}")

# Example usage
video_path = r"C:\Users\chinm\OneDrive\Desktop\anomaly\Explosion\Explosion006_x264.mp4"



import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image      
import os

# Define paths
model_name = "csr2000/UCF_Crime"
video_frames_path = r"C:\Users\kavit\OneDrive\Desktop\chinmayee1\BOLT AI (3)\BOLT AI\backend\processedvideos" # Folder containing video frames

# Load the model and feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained(model_name)

# Preprocessing parameters (from the provided JSON)
normalize = feature_extractor.do_normalize
resize = feature_extractor.do_resize         
 
# Handle size extraction (ensure it works if size is a dict or int)
if isinstance(feature_extractor.size, dict):
    image_size = (feature_extractor.size.get("height", 224), feature_extractor.size.get("width", 224))
else:
    image_size = (feature_extractor.size, feature_extractor.size)

image_mean = feature_extractor.image_mean
image_std = feature_extractor.image_std

# Function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    if resize:
        image = image.resize(image_size)

    image = torch.tensor(feature_extractor(image, return_tensors="pt")["pixel_values"][0])

    return image.unsqueeze(0)  # Add batch dimension

# Iterate through frames and predict anomaly class
frame_predictions = []
for frame in sorted(os.listdir(video_frames_path)):
    frame_path = os.path.join(video_frames_path, frame)

    if not frame.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    # Preprocess the frame
    input_tensor = preprocess_image(frame_path)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = outputs.logits.argmax(dim=1).item()
        frame_predictions.append(predicted_class)

# Print or save the predictions
print("Predictions for each frame:", frame_predictions)

# (Optional) Aggregate predictions to make a decision for the entire video
video_prediction = max(set(frame_predictions), key=frame_predictions.count)
print("Predicted anomaly class for the video:", video_prediction)

