import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import os
import numpy as np
import tempfile
import io

# UCF Crime dataset class mapping
LABELS_MAP = {
    0: "Abuse",
    1: "Arrest",
    2: "Arson",
    3: "Assault",
    4: "Burglary",
    5: "Explosion",
    6: "Fighting",
    7: "NormalVideos",
    8: "RoadAccidents",
    9: "Robbery",
    10: "Shooting",
    11: "Shoplifting",
    12: "Stealing",
    13: "Vandalism"
}

def detect_video_anomaly(file_storage):
    """
    Process a video file from Flask's FileStorage and detect anomalies.
    
    Args:
        file_storage: Flask FileStorage object containing the video
        
    Returns:
        dict: Contains predictions and confidence scores
    """
    
    # Create a temporary directory to store frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded file to a temporary file
        video_bytes = file_storage.read()
        temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
        
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)
        
        # Extract frames
        frames = preprocess_video_to_frames(temp_video_path, temp_dir)
        
        # Load the model and feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        model = AutoModelForImageClassification.from_pretrained("csr2000/UCF_Crime")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Process frames and get predictions
        frame_predictions = process_frames(frames, feature_extractor, model, device)
        
        # Calculate confidence scores
        prediction_counts = {}
        for pred in frame_predictions:
            label = LABELS_MAP.get(pred, str(pred))
            prediction_counts[label] = prediction_counts.get(label, 0) + 1
            
        total_frames = len(frame_predictions)
        confidence_scores = {
            label: (count / total_frames) * 100 
            for label, count in prediction_counts.items()
        }
        
        # Get most common prediction as overall prediction
        most_common_pred = max(set(frame_predictions), key=frame_predictions.count)
        overall_prediction = LABELS_MAP.get(most_common_pred, str(most_common_pred))
        
        # Convert numeric predictions to labels
        frame_predictions = [LABELS_MAP.get(pred, str(pred)) for pred in frame_predictions]
        
        return {
            'overall_prediction': overall_prediction,
            'frame_predictions': frame_predictions,
            'confidence_scores': confidence_scores
        }

def preprocess_video_to_frames(video_path, output_dir, num_frames=50):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Unable to open video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)
    
    frame_count = 0
    extracted_frames = 0
    frame_paths = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{extracted_frames:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_frames += 1
            
            if extracted_frames >= num_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return frame_paths

def process_frames(frame_paths, feature_extractor, model, device):
    """Process all frames and return predictions"""
    frame_predictions = []
    
    for frame_path in frame_paths:
        # Preprocess the frame
        image = Image.open(frame_path).convert("RGB")
        
        # Prepare image for model
        inputs = feature_extractor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class = outputs.logits.argmax(dim=1).item()
            frame_predictions.append(predicted_class)
    
    return frame_predictions