🔥 Skin Burn Severity Detection using YOLO

This project focuses on detecting and classifying skin burn severity (first, second, and third degree) using deep learning with YOLO.
The model was trained on annotated burn images and deployed using OpenCV’s DNN module with ONNX.

Achieved 94% accuracy in burn severity classification, enabling faster and more reliable medical triaging.

📌 Project Highlights

✅ Built a YOLO-based object detection & classification model for skin burns.

✅ Trained and validated on a dataset of 1,400+ annotated images.

✅ Achieved 94% accuracy in distinguishing first, second, and third-degree burns.

✅ Deployed YOLO model with OpenCV DNN (ONNX format) for lightweight inference.

✅ Designed a modular Python script for prediction and visualization.

📂 Project Structure
skin-burn-severity-detection-

```skin-burn-severity-detection/
│
├── models/                      # Trained YOLO model (.onnx)
├── data.yaml                    # Dataset config file (class labels, nc, paths)
├── utils/                       # Helper scripts (optional)
├── predictions.py               # Script for running YOLO predictions
├── skin_burn_detection.ipynb    # Training + evaluation notebook
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview (this file)```


🧠 Model Details

Model: YOLOv5 (exported to ONNX for deployment)

Classes:

0 → First Degree Burn

1 → Second Degree Burn

2 → Third Degree Burn

Frameworks/Libraries:

PyTorch (training)

OpenCV (inference with ONNX)

NumPy, PyYAML

⚙️ Code Example
import cv2
from predictions import YOLO_Pred

# Initialize YOLO predictor
yolo = YOLO_Pred("models/skin_burn.onnx", "data.yaml")

# Load an image
image = cv2.imread("sample_burn.jpg")

# Run prediction
detections = yolo.predictions(image)

# Show result
cv2.imshow("Skin Burn Severity Detection", image)
cv2.waitKey(0)

📊 Results

Training Accuracy: 94%

Validation Loss: Low generalization error observed

Inference Speed: Real-time detection using CPU with OpenCV DNN

📸 Sample Predictions

(Insert screenshots here of detected burns with bounding boxes + class labels)

▶️ Run Locally

Clone the repository:

git clone https://github.com/your-username/skin-burn-severity-detection.git
cd skin-burn-severity-detection


Install dependencies:

pip install -r requirements.txt


Run prediction:

python predictions.py --image sample_burn.jpg

🛠️ Tech Stack

Deep Learning: YOLOv5 → ONNX

Inference: OpenCV DNN

Languages: Python

Libraries: numpy, pyyaml, opencv-python
