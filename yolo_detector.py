import os
import sys
import torch
import cv2

from logging_code import setup_logging

logger = setup_logging("YOLO_DETECTOR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_DET_PATH = os.path.join(BASE_DIR, "yolov7_detection")

sys.path.insert(0, YOLO_DET_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression


COLORS = {
    "person": (124, 255, 156),
    "car": (255, 230, 78),
    "bike": (56, 255, 255),
    "fire": (255, 0, 255),
    "smoke": (150, 200, 128),
    "helmet": (255, 155, 100),
    "jacket": (155, 190, 255)
}


class YOLODetector:

    def __init__(
        self,
        weights_path,
        device="cpu"
    ):

        self.device = device
        self.model = None

        try:

            logger.info(f"Loading Detection Model : {weights_path}")

            self.model = attempt_load(
                weights_path,
                map_location=device
            )

            self.model.eval()

            logger.info(
                "YOLOv7 Detection Loaded Successfully"
            )

        except Exception as e:

            logger.error(f"Detection Load Error : {e}")

    def process_frame(self, frame):

        try:

            if self.model is None:

                logger.error("Detection model not loaded")

                return frame

            original = frame.copy()

            img = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img_tensor = torch.from_numpy(img).to(self.device)

            img_tensor = (
                img_tensor.permute(2, 0, 1)
                .float() / 255.0
            )

            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():

                pred = self.model(img_tensor)[0]

            pred = non_max_suppression(
                pred,
                0.25,
                0.45
            )[0]

            if pred is None:

                return original

            for *xyxy, conf, cls in pred:

                x1, y1, x2, y2 = map(int, xyxy)

                label_name = self.model.names[int(cls)]

                color = COLORS.get(
                    label_name,
                    (0, 255, 0)
                )

                label = f"{label_name} {conf:.2f}"

                cv2.rectangle(
                    original,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    original,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            return original

        except Exception as e:

            logger.error(f"Detection Process Error : {e}")

            return frame