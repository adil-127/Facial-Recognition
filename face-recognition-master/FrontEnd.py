import time
import cv2
import requests
import json
import base64
from face_detection.yolov5_face.detector import Yolov5Face

# Initialize the face detector
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Initialize variables for measuring frame rate
    start = time.time_ns()
    frame_count = 0
    fps = -1
    count=0
    frames=[]
    # Read frames from the camera
    caption='Processing'

    while True:
        # Capture a frame from the camera
        count+=1
        ret, frame = cap.read()
        if not ret:
            break

        # Get faces and landmarks using the face detector
        bboxes, landmarks = detector.detect(image=frame)

        # Prepare data for sending to the backend
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        landmarks_list = [[{'x': int(point[0]), 'y': int(point[1])} for point in face_landmarks] for face_landmarks in landmarks]
        frames.append(frame_base64)
        print(len(frames))
        if count%50==0:
            data = {
                'frame': frames,
                'landmarks': landmarks_list
            }
        # Send data to the backend
            try:
                response = requests.post("http://127.0.0.1:8000/process-frame/", json=data)
                response.raise_for_status()
                try:
                    caption=response.json()["classification"]
                    print(response.json()["classification"])
                    frames=[]
                    data=None

                except json.JSONDecodeError:
                    print("Failed to decode JSON response:", response.text)
                    frames=[]
                    data=None
            except requests.RequestException as e:
                print("Request failed:", e)
                frames=[]
                data=None
                
                
              
        # Draw bounding boxes and landmarks on the frame
        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
        
        for face_landmarks in landmarks:
            for id, key_point in enumerate(face_landmarks):
                cv2.circle(frame, tuple(key_point), 1, (0, 255, 0), -1)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Get the latest caption from the backend
        # try:
        #     caption_response = requests.get("http://127.0.0.1:8000/latest-caption/")
        #     caption_response.raise_for_status()
        #     caption_data = caption_response.json()
        #     latest_caption = caption_data.get("latest_caption", "No caption available")
        # except requests.RequestException as e:
        #     latest_caption = "Error fetching caption"
        #     print("Request failed:", e)

        # Display the latest caption on the frame
        cv2.putText(frame, caption, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the result in a window
        cv2.imshow("Face Detection", frame)

        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release video and camera, and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
