from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark
import os
from pathlib import Path
import tensorflow as tf
import time


model = YOLO("yolov8n.pt")

results = model.train(
    data=r"D:\VScode_projects\proyecto_TFOD_YOLO\config\config.yaml",
    epochs=1,
    imgsz=640,
    batch=16,
    name="yolov8n_detection",
    project="runs/detect",
    device="cpu",
    patience=50,
    save=True,
    plots=True
)

#Validacion y metricas mAP
metrics = model.val()
print("================================")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print("================================")


