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


#Velocidad de inferencia
best_model_path = "runs/detect/yolov8n_detection/weights/best.pt"

print("================================")
print("Benchmarking...")
print("================================")

# Fix OpenVINO errors
benchmark_results = benchmark(
    model=best_model_path,
    data=r"D:\VScode_projects\proyecto_TFOD_YOLO\config\config.yaml",
    imgsz=640,
    half=False,
    device="cpu",
    verbose=True
)

#Tamano del modelo
model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
print("================================")
print(f"Tamano del modelo: {model_size_mb:.4f}")
print("================================")


#Exportacion a diferentes formatos
best_model = YOLO(best_model_path)

# ONNX
onnx_path = best_model.export(format="onnx", dynamic=True, simplify=True)
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
print("================================")
print(f"ONNX: {onnx_size:.4f}")
print("================================")


# TensorRT
trt_path = best_model.export(format="engine", half=True, workspace=4)
trt_size = os.path.getsize(trt_path) / (1024 * 1024)
print("================================")
print(f"TensorRT: {trt_size:.4f}")
print("================================")
