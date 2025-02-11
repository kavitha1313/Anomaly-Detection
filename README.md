# Anomaly Detection in Surveillance Videos

## Overview

This project focuses on **anomaly detection in surveillance videos** using the **Vision Transformer (ViT) model**. The system aims to identify unusual activities in real-time video streams or pre-recorded footage, enhancing security and surveillance applications.

## Features

- Utilizes a **pretrained ViT model** for feature extraction.
- Processes surveillance video frames for anomaly detection.
- Detects and flags unusual activities in real-time.
- Scalable for various surveillance environments.

## Technologies Used

- **Python** (for backend processing)
- **TensorFlow / PyTorch** (for implementing ViT)
- **OpenCV** (for video frame processing)
- **Flask / FastAPI** (for serving the model via API)
- **Frontend Framework** (React/HTML for UI if applicable)

## Installation

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd anomaly-detection-surveillance
```

### 2. Install Dependencies

#### **Backend Dependencies**

```sh
pip install -r requirements.txt
```

#### **Frontend Dependencies** (if applicable)

```sh
cd frontend
npm install
```

> **Note:** The `node_modules` folder is not included in the repository. You must run `npm install` inside the `frontend` folder to install frontend dependencies.

## Usage

### **Run Backend Server**

```sh
python backend/server.py
```

### **Run Frontend**

```sh
cd frontend
npm start
```

## Dataset

- We used the **UCF-Crime dataset** for training and testing.
- Ensure the dataset is placed in the correct directory before running the model.

## Contribution

Feel free to fork this repository and submit pull requests if you have improvements.

## License

This project is open-source and available under the **MIT License**.


