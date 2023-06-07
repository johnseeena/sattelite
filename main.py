from IPython import display
import torch
import ultralytics
from ultralytics import YOLO


def yolo_training():
    display.clear_output()
    ultralytics.checks()

    model = YOLO("yolov8m.yaml")
    if torch.cuda.is_available():
        result = model.train(data="dataset.yaml", epochs=100, imgsz=640, device=0, save=True)
    else:
        result = model.train(data="dataset.yaml", epochs=100, imgsz=640, device='cpu', save=True)


if __name__ == '__main__':
    yolo_training()
