import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from time import time
import os

# Email Alert Setup
def setup_email():
    """Set up the SMTP server and email login details."""
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login("prithvireddy27945@gmail.com", "ofuj uqve ohre dvab")  # Use app password
    return server

def send_email(server, to_email, from_email, snapshot_path, object_detected=1):
    """Sends an email notification with an attached snapshot."""
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = "Security Alert"

    # Email body
    message_body = f"ALERT - {object_detected} object(s) detected!"
    message.attach(MIMEText(message_body, "plain"))

    # Attach the snapshot
    with open(snapshot_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename=intruder_snapshot.jpg",
        )
        message.attach(part)

    server.sendmail(from_email, to_email, message.as_string())

class ObjectDetection:
    def __init__(self, capture_index, server):
        self.capture_index = capture_index
        self.email_sent = False
        self.model = YOLO("yolo11n.pt")  # Path to YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.server = server

        # Load the face recognizer model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer/trainer2.yml")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def predict(self, im0):
        """Run prediction using the YOLO model for the input image."""
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        """Displays FPS on the image frame."""
        end_time = time()
        fps = 1 / (end_time - self.start_time)
        text = f"FPS: {int(fps)}"
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    def plot_bboxes(self, results, im0):
        """Draws bounding boxes for detected objects in the frame."""
        class_ids = []
        annotator = Annotator(im0, 3, results[0].names)
        for box, cls in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.cls.cpu().tolist()):
            if cls == 0:  # Assuming class ID 0 is for people
                annotator.box_label(box, label="Person", color=colors(cls, True))
                class_ids.append(cls)

                # Extract face ROI for recognition
                x1, y1, x2, y2 = map(int, box)
                face_roi = im0[y1:y2, x1:x2]
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

                # Detect and recognize face
                faces = self.face_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
                for (fx, fy, fw, fh) in faces:
                    id_, confidence = self.recognizer.predict(gray_face[fy:fy+fh, fx:fx+fw])
                    if confidence < 50:  # Recognition threshold
                        annotator.box_label(box, label=f"Owner ({confidence:.2f})", color=(0, 255, 0))
                    else:
                        annotator.box_label(box, label=f"Intruder ({confidence:.2f})", color=(0, 0, 255))
        return im0, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(1)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            self.start_time = time()
            ret, im0 = cap.read()
            assert ret
            results = self.predict(im0)
            im0, class_ids = self.plot_bboxes(results, im0)

            # Send email if an intruder is detected and email has not been sent
            if len(class_ids) > 0 and not self.email_sent:
                snapshot_path = "intruder_snapshot.jpg"
                cv2.imwrite(snapshot_path, im0)  # Save snapshot
                send_email(self.server, "prithvireddy27945@gmail.com", "prithvireddy27945@gmail.com", snapshot_path, len(class_ids))
                self.email_sent = True

            self.display_fps(im0)
            cv2.imshow("YOLO Detection and Face Recognition", im0)
            if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        self.server.quit()

# Main Execution
if __name__ == "__main__":
    server = setup_email()  # Set up email server
    detector = ObjectDetection(capture_index=0, server=server)  # Initialize Object Detection
    detector()  # Run the detection
