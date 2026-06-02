import os
import sys
import torch
import cv2
import numpy as np

from logging_code import setup_logging

logger = setup_logging("YOLO_SEGMENTOR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================
# YOLOv7 SEG PATHS
# =========================

YOLO_SEG_PATH = os.path.join(
    BASE_DIR,
    "yolov7_seg"
)

SEGMENT_UTILS_PATH = os.path.join(
    YOLO_SEG_PATH,
    "utils",
    "segment"
)

# =========================
# ADD PATHS
# =========================

sys.path.insert(0, YOLO_SEG_PATH)

sys.path.insert(0, SEGMENT_UTILS_PATH)

# =========================
# IMPORTS
# =========================

from models.experimental import attempt_load

from utils.general import non_max_suppression

from general import process_mask


class YOLOSegmentor:

    def __init__(
        self,
        weights_path,
        device="cpu"
    ):

        self.device = device

        self.model = None

        try:

            logger.info(
                f"Loading Segmentation Model : {weights_path}"
            )

            self.model = attempt_load(
                weights_path,
                map_location=device
            )

            self.model.eval()

            logger.info(
                "YOLOv7 Segmentation Loaded Successfully"
            )

        except Exception as e:

            logger.error(
                f"Segmentation Load Error : {e}"
            )

    def process_frame(self, frame):

        try:

            if self.model is None:

                logger.error(
                    "Segmentation model not loaded"
                )

                return frame

            original = frame.copy()

            img = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            img = cv2.resize(
                img,
                (640, 640)
            )

            img_tensor = torch.from_numpy(
                img
            ).to(self.device)

            img_tensor = (
                img_tensor.permute(2, 0, 1)
                .float() / 255.0
            )

            img_tensor = img_tensor.unsqueeze(0)

            with torch.no_grad():

                pred, out = self.model(
                    img_tensor
                )

            proto = out[1]

            pred = non_max_suppression(
                pred,
                0.25,
                0.45,
                classes=[0]
            )

            if pred[0] is None:

                return original

            det = pred[0]

            masks = process_mask(
                proto[0],
                det[:, 6:],
                det[:, :4],
                img_tensor.shape[2:],
                upsample=True
            )

            for mask in masks:

                mask = mask.cpu().numpy()

                mask = cv2.resize(
                    mask,
                    (
                        original.shape[1],
                        original.shape[0]
                    )
                )

                colored_mask = np.zeros_like(
                    original
                )

                colored_mask[:, :, 1] = (
                    mask > 0.5
                ) * 255

                original = cv2.addWeighted(
                    original,
                    1,
                    colored_mask,
                    0.5,
                    0
                )

            return original

        except Exception as e:

            logger.error(
                f"Segmentation Process Error : {e}"
            )

            return frame