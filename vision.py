from ultralytics import YOLO
from helper import device
import cv2 as cv
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from summary import Summary
from helper import load_json_variable
from PIL import Image
import csv
import os
from alert_system import AlertSystem

logging.basicConfig(level=logging.INFO)

class Vision:
    def __init__(self, model_name: str, confidence_threshold: float = 0.6, stream: bool = False, summary_model: Summary = None):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.MODEL = self.load_model()
        self.stream = stream
        self.summary = summary_model
        self.alert_system = AlertSystem()


    def load_model(self):
        MODEL = YOLO(self.model_name, verbose=False)
        # MODEL = MODEL.to(device())
        return MODEL

    def append_to_csv(self, summary_text: str, detections: List[Dict], object_counts: Dict, alert_data: Dict = None):
        csv_filename = load_json_variable("csv_filename")
        timestamp = datetime.now().isoformat()
        
        file_exists = os.path.isfile(csv_filename)
        
        try:
            with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'summary', 'detections_json', 'object_counts_json', 'total_objects', 
                             'alert_status', 'alert_severity', 'alert_type', 'alert_message', 'alert_details_json']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    print(f"Created new CSV file: {csv_filename}")
                
                alert_status = 'false'
                alert_severity = 'none'
                alert_type = 'none'
                alert_message = 'No alerts'
                alert_details_json = '{}'
                
                if alert_data:
                    alert_status = 'true' if alert_data.get('should_alert', False) else 'false'
                    alert_severity = alert_data.get('severity', 'none')
                    alert_type = alert_data.get('alert_type', 'none')
                    alert_message = alert_data.get('message', 'No alerts')
                    alert_details_json = json.dumps(alert_data.get('details', {}))
                
                writer.writerow({
                    'timestamp': timestamp,
                    'summary': summary_text,
                    'detections_json': json.dumps(detections),
                    'object_counts_json': json.dumps(object_counts),
                    'total_objects': len(detections),
                    'alert_status': alert_status,
                    'alert_severity': alert_severity,
                    'alert_type': alert_type,
                    'alert_message': alert_message,
                    'alert_details_json': alert_details_json
                })
                
                print(f"Summary and alert data appended to {csv_filename} at {timestamp}")
                if alert_data and alert_data.get('should_alert', False):
                    print(f"ALERT: {alert_message}")
                
        except Exception as e:
            print(f"Error writing to CSV: {e}")


    def process_viewpoint(self, source):
        if isinstance(source, str):
            if source.endswith(('.jpg', '.jpeg', '.png')):
                image = cv.imread(source)
                self.detect_objects(image)
            elif source.endswith(('.mp4', '.avi')):
                self.process_video(source)
            elif source.startswith(('tcp://', 'udp://', 'rtsp://', 'rtmp://', 'http://', 'https://')):
                self.process_video(source)
            else:
                logging.error("Unsupported file format.")
        else:
            self.process_video(source)

    def process_video(self, video_source):
        cap = cv.VideoCapture(video_source)


        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FPS, 30)

        all_detections = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.detect_objects(frame)

            if self.stream:
                cv.imshow("VisionSense Live Stream", frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv.destroyAllWindows()

    def detect_objects(self, image):
        print("In vision object detecion")
        print()

        try:
            results = self.MODEL(image)
            detections = []
            object_counts = {}
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.MODEL.names[class_id]
                        print(datetime.now().isoformat())
                        if confidence >= self.confidence_threshold:
                            detection = {
                                "timestamp": datetime.now().isoformat(),
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": round(confidence, 3),
                                "bbox": [x1, y1, x2, y2],
                                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                                "area": (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
                            
                            object_counts[class_name] = object_counts.get(class_name, 0) + 1

                            print(object_counts)
                            print(detections)
                            if detections and image is not None:
                                pil_image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                                summary_text = self.summary.generate_summary(str(detections), len(detections), pil_image)
                                print("Generated Summary:", summary_text)
                                
                                timestamp = datetime.now().isoformat()
                                alert_data = self.alert_system.analyze_detections(detections, timestamp)
                                
                                if alert_data.get('should_alert', False):
                                    print("ALERT TRIGGERED:")
                                    print(self.alert_system.format_alert_message(alert_data))
                                
                                self.append_to_csv(summary_text, detections, object_counts, alert_data)
                            else:
                                print("No detections or image to summarize.")

                            # self.summary.generate_summary(detections, object_counts, image)
    
                            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {confidence:.2f}"
                            cv.putText(
                                image,
                                label,
                                (x1, y1 - 10),
                                cv.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )

            if self.stream:
                cv.imshow("VisionSense Live Stream", image)
        except Exception as e:
            logging.error(f"Error in object detection: {e}")
        return None

