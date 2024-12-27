import cv2
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta

# Path configurations
TRAINED_MODEL_PATH = "trainer/trainer4.yml"  # Trained LBPH model
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
RUNTIME_FACE_DIR = "runtime_faces"

# Create directory to save runtime faces for debugging
os.makedirs(RUNTIME_FACE_DIR, exist_ok=True)

# Load the trained face recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINED_MODEL_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Owner ID and label
owner_id = 1  # Replace with the ID you assigned to the owner during training
owner_label = "Owner"

# Email Alert Setup
EMAIL_FROM = "prithvireddy27945@gmail.com"  # Your email
EMAIL_TO = "prithvireddy27945@gmail.com"  # Recipient email
EMAIL_PASSWORD = "ofuj uqve ohre dvab"  # App password

def setup_email():
    """Set up the SMTP server and email login details."""
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(EMAIL_FROM, EMAIL_PASSWORD)
    return server

def send_email(server, subject, body, attachment_path=None):
    """Sends an email notification with optional attachment."""
    message = MIMEMultipart()
    message["From"] = EMAIL_FROM
    message["To"] = EMAIL_TO
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    if attachment_path:
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        message.attach(part)

    server.sendmail(EMAIL_FROM, EMAIL_TO, message.as_string())

# Face Recognition and Security System
def recognize_faces():
    """Recognizes faces live from the webcam and sends an email for unrecognized faces."""
    server = setup_email()
    cap = cv2.VideoCapture(1)  # Open webcam
    if not cap.isOpened():
        print("[ERROR] Could not open the webcam.")
        return

    email_sent = False
    # timeout = timedelta(minutes=5)
    # last_taken = datetime.now() - timeout 
    while True:
        
        time.sleep(0.3)
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Convert the frame to grayscale for face recognition
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.2, 
            minNeighbors=8, 
            minSize=(50, 50)
        )

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = gray_frame[y:y + h, x:x + w]

            # Recognize the face
            id_, confidence = recognizer.predict(face_roi)

            if confidence > 70:  # Recognized as the owner
                label = f"{owner_label} ({confidence:.2f})"
                color = (0, 255, 0)  # Green
            else:  # Unrecognized face
                label = f"Unknown ({confidence:.2f})"
                color = (0, 0, 255)  # Red

                # Save snapshot and send email if not already sent
                if not email_sent: # and datetime.now() - last_taken >= timeout:
                    snapshot_path = os.path.join(RUNTIME_FACE_DIR, f"intruder_{x}_{y}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    send_email(server, "Security Alert: Intruder Detected", "An unknown person was detected!", snapshot_path)
                    print("[INFO] Email sent with intruder snapshot.")
                    email_sent = True
                    # last_taken = datetime.now()

            # Draw a rectangle around the face and display the label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the live video feed with annotations
        cv2.imshow("Face Recognition", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    server.quit()

if __name__ == "__main__":
    print("[INFO] Starting face recognition...")
    recognize_faces()
