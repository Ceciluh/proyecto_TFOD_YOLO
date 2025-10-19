from ultralytics import YOLO
import cv2

model = YOLO('best.onnx')

video_path = r'D:\VScode_projects\proyecto_TFOD_YOLO\data\test\test_5.mp4'

cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = 'output_detection.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    annotated_frame = results[0].plot()
    
    out.write(annotated_frame)
    
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    
    cv2.imshow('YOLOv8 Detection', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()