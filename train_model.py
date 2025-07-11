from ultralytics import YOLO
# Load model
model = YOLO("yolov8m.pt")
# Path to data.yaml
yaml_path = r"D:\PROJECTS in B.Tech\Brain Tumour Detection Using_Python\data\data.yaml"
# Train model
model.train(
    data=yaml_path,
    epochs=80,
    imgsz=224,
    batch=8,
    project="brain_tumour_detector",
    name="yolov8m_128img_100epoch",
    workers=1,
)