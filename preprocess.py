import os
import cv2

def extract_frames(video_path, output_dir, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % frame_rate == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
            cv2.imwrite(output_path, frame)
            frame_id += 1

    cap.release()

def preprocess_videos(dataset_dir, output_dir, frame_rate=1):
    for split in ['train', 'val']:
        input_split_dir = os.path.join(dataset_dir, split)
        output_split_dir = os.path.join(output_dir, split)
        
        for class_name in os.listdir(input_split_dir):
            input_class_dir = os.path.join(input_split_dir, class_name)
            output_class_dir = os.path.join(output_split_dir, class_name)

            for video_file in os.listdir(input_class_dir):
                video_path = os.path.join(input_class_dir, video_file)
                extract_frames(video_path, output_class_dir, frame_rate)

# Run preprocessing
if __name__ == "__main__":
    preprocess_videos("dataset", "frames", frame_rate=10)
