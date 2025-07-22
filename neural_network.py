import os
from ultralytics import YOLO


model = None
model_path = 'yolov8x/best.pt'


def init_neural_model():
    global model
    pid = os.getpid()
    list_info = []

    try:

        model = YOLO(model_path)
        list_info.append(f'PID={pid} model successfully loaded')
    except Exception:
        list_info.append(f'PID={pid} failed loading model')


def process_input(file, content_type):
    pid = os.getpid()
    res = []
    if model is None:
        init_neural_model()

    try:
        results = model.predict(
            source=file,
            imgsz=640,
            conf=0.25,
            save=True,
            project='runs',
            name="result",
            exist_ok=True,
        )
        if content_type.startswith("image/"):
            for r in results:
                if len(r.boxes.cls) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = r.names[cls_id]
                        res.append(f"- {cls_name} с уверенностью {box.conf[0]:.2f}")
                else:
                    res.append("Классы не обнаружены")
        elif content_type.startswith("video/"):
            found = any(len(r.boxes) > 0 for r in results)
            if found:
                res.append("Объект обнаружен хотя бы в одном кадре")
            else:
                res.append("Никаких объектов не обнаружено")
        return pid, res, results, 'Successfully processed'
    except Exception as e:
        return pid, 'Error', None, f"PID={pid} Processing failed: {e}"

