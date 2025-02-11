from torchvision import transforms, models
import torch
from PIL import Image

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(ANOMALY_CLASSES))
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Inference
def classify_frame(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return ANOMALY_CLASSES[predicted.item()]

# Example
print(classify_frame("path_to_frame.jpg"))
