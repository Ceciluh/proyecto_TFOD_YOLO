#NOTA: este fue el codigo usado antes de transferirlo a google colab, 
from ultralytics import YOLO
import os
import time

model = YOLO("yolov8s.pt")

# Training
results = model.train(
    data=r"D:/VScode_projects/proyecto_TFOD_YOLO/config/config.yaml",
    epochs=150,
    imgsz=640,
    batch=32,
    name="yolov8s_detection",
    project=r"D:/VScode_projects/proyecto_TFOD_YOLO/runs/detect",
    device="cpu",  
    save=True,
    plots=True
)

metrics = model.val()
print("============ Metricas ==============")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

# Inference speed test
print("============ Velocidad de inferencia ==============")

best_model_path = r"D:/VScode_projects/proyecto_TFOD_YOLO/runs/detect/yolov8s_detection/weights/best.pt"
model = YOLO(best_model_path)

test_image_path = r"D:/VScode_projects/proyecto_TFOD_YOLO/data/images/val/office_objects_59.jpg"

for i in range(10):
    model(test_image_path, verbose=False)

inference_times = []
for i in range(100):
    start_time = time.time()
    results = model(test_image_path, verbose=False)
    end_time = time.time()
    inference_times.append((end_time - start_time) * 1000)

avg_time = sum(inference_times) / len(inference_times)
print(f"Tiempo promedio: {avg_time:.4f} ms")

print("=========== Tamaño del modelo ===============")
model_size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
print(f"Tamaño del modelo: {model_size_mb:.4f} MB")
print("=============================================")

model.export(format='onnx', half=True)
