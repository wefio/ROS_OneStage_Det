from ultralytics import YOLO

# Load a model
model = YOLO(model="models/yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model.track(source="只因你太美（鸡你太美）原版 - 1.只因你太美（鸡你太美）原版(Av51818204,P1).mp4",show=True,save=True)  # predict on an image