import sys
import cv2
import json
import pika
import base64
import numpy as np
import redis
import warnings
warnings.filterwarnings("ignore")
from config import (rabbitmq,detection_queues,routing_keys,redis_server)
from logging_code import setup_logging
logger = setup_logging("ROI")


class ROIConsumer:

    def __init__(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq["host"],port=rabbitmq["port"]))
        self.channel = self.connection.channel()


        self.channel.exchange_declare(exchange=rabbitmq["detection_exchange"],exchange_type=rabbitmq["exchange_type"],durable=False)

        self.channel.queue_declare(queue=detection_queues["vehicle"],durable=True)
        self.channel.queue_bind(exchange=rabbitmq["detection_exchange"],queue=detection_queues["vehicle"],routing_key=routing_keys["vehicle"])

        self.channel.queue_declare(queue=detection_queues["fire_smoke"],durable=True)
        self.channel.queue_bind(exchange=rabbitmq["detection_exchange"],queue=detection_queues["fire_smoke"],routing_key=routing_keys["fire_smoke"])

        self.channel.queue_declare(queue=detection_queues["safety"],durable=True)
        self.channel.queue_bind(exchange=rabbitmq["detection_exchange"],queue=detection_queues["safety"],routing_key=routing_keys["safety"])

        logger.info("RabbitMQ Connected")

        self.redis_db = redis.StrictRedis(host=redis_server["host"],port=redis_server["port"],db=redis_server["db"],decode_responses=False)
        logger.info("Redis Connected")

        self.roi_coords = {}

    def get_fixed_roi(self, width, height):

        roi_width = int(width * 0.8)
        roi_height = int(height * 0.9)

        center_x = width // 2
        center_y = height // 2

        x1 = center_x - roi_width // 2
        y1 = center_y - roi_height // 2

        x2 = center_x + roi_width // 2
        y2 = center_y + roi_height // 2

        return x1, y1, x2, y2


    def is_inside_roi(self, bbox, roi):

        bx1, by1, bx2, by2 = bbox
        rx1, ry1, rx2, ry2 = roi

        # Center point of detection
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        return (
                rx1 <= cx <= rx2 and
                ry1 <= cy <= ry2
        )


    def save_frame_to_redis(self, redis_key, frame):

        _, buffer = cv2.imencode(".jpg",frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        self.redis_db.rpush(redis_key,frame_b64)

        logger.info(f"Saved To Redis : {redis_key}")

    def process_message(self,ch,method,properties,body):

        try:

            data = json.loads(body.decode("utf-8"))
            meta = data["meta"]
            detections = meta["detections"]
            cam_id = meta["cam_id"]
            logger.info(
                f"MESSAGE RECEIVED FROM {cam_id} | DETECTIONS={len(detections)}"
            )
            frame_b64 = data["frame_b64"]

            frame_bytes = base64.b64decode(frame_b64)
            frame = cv2.imdecode(np.frombuffer(frame_bytes,np.uint8),cv2.IMREAD_COLOR)

            if frame is None:
                return

            h, w, _ = frame.shape

            if cam_id not in self.roi_coords:

                self.roi_coords[cam_id] = \
                    self.get_fixed_roi(w, h)
            rx1, ry1, rx2, ry2 = \
                self.roi_coords[cam_id]

            cv2.rectangle(frame,(rx1, ry1),(rx2, ry2),(0, 0, 255),3)
            cv2.putText(frame,"ROI",(rx1, ry1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 0, 255),2)

            for det in detections:
                label = det["class"]
                confidence = det["confidence"]
                bbox = det["bbox"]
                x1, y1, x2, y2 = bbox

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                cv2.putText(frame,f"{label} {confidence:.2f}",(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2)

                inside = self.is_inside_roi(
                    (x1, y1, x2, y2),
                    (rx1, ry1, rx2, ry2))

                logger.info(
                    f"CAM={cam_id} LABEL={label} ROI={inside}"
                )

                color = (0, 255, 0) if inside else (0, 0, 255)

                cv2.putText(
                    frame,
                    f"ROI:{inside}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if inside:

                    logger.info(
                        f"CAM={cam_id} LABEL={label} ROI={inside}"
                    )

                    if label in ["car", "bike"]:
                        self.save_frame_to_redis(
                            "vehicle_frames",
                            frame
                        )

                    elif label in ["fire", "smoke"]:
                        self.save_frame_to_redis(
                            "fire_smoke_frames",
                            frame
                        )

                    elif label in ["helmet", "jacket"]:
                        self.save_frame_to_redis(
                            "safety_frames",
                            frame
                        )

            display_frame = cv2.resize(frame,(500,400))
            cv2.imshow(f"ROI - {cam_id}",display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

        except Exception as e:
            logger.info(f"ERROR : {e}")

    def start(self):

        logger.info("Registering vehicle consumer")
        self.channel.basic_consume(
            queue=detection_queues["vehicle"],
            on_message_callback=self.process_message,
            auto_ack=True
        )

        logger.info("Registering fire consumer")
        self.channel.basic_consume(
            queue=detection_queues["fire_smoke"],
            on_message_callback=self.process_message,
            auto_ack=True
        )

        logger.info("Registering safety consumer")
        self.channel.basic_consume(
            queue=detection_queues["safety"],
            on_message_callback=self.process_message,
            auto_ack=True
        )

        logger.info("Waiting For Frames...")
        self.channel.start_consuming()

if __name__ == "__main__":

    obj = ROIConsumer()
    obj.start()