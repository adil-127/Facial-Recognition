import time

import cv2

from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face

# Initialize the face detector
detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")


def main():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Initialize variables for measuring frame rate
    start = time.time_ns()
    frame_count = 0
    fps = -1

    # Save video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    video = cv2.VideoWriter(
        "results/face-detection.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size
    )

    # Read frames from the camera
    while True:
        # Capture a frame from the camera
        _, frame = cap.read()

        # Get faces and landmarks using the face detector
        bboxes, landmarks = detector.detect(image=frame)
        h, w, c = frame.shape

        tl = 1 or round(0.002 * (h + w) / 2) + 1  # Line and font thickness
        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # Draw bounding boxes and landmarks on the frame
        for i in range(len(bboxes)):
            # Get location of the face
            x1, y1, x2, y2, score = bboxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)

            # Draw facial landmarks
            for id, key_point in enumerate(landmarks[i]):
                cv2.circle(frame, tuple(key_point), tl + 1, clors[id], -1)

        # Calculate and display the frame rate
        frame_count += 1
        if frame_count >= 30:
            end = time.time_ns()
            fps = 1e9 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        # Save the frame to the video
        video.write(frame)

        # Show the result in a window
        cv2.imshow("Face Detection", frame)

        # Press 'Q' on the keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Release video and camera, and close all OpenCV windows
    video.release()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == "__main__":
    main()




# import cv2
# import numpy as np
# from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face

# # Initialize the face detector
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
# # detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# # Define a function to compute the Euclidean distance between two points
# def euclidean_distance(pt1, pt2):
#     return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# # Define a threshold for movement detection
# MOVEMENT_THRESHOLD = 5  # Adjust as needed

# def main():
#     # Open the camera
#     cap = cv2.VideoCapture(0)

#     # Initialize variables
#     prev_landmarks = None

#     # Read frames from the camera
#     while True:
#         # Capture a frame from the camera
#         _, frame = cap.read()

#         # Get faces and landmarks using the face detector
#         bboxes, landmarks = detector.detect(image=frame)
        
#         # Check if any faces are detected
#         if len(landmarks) > 0:
#             # Check for movement between consecutive frames
#             if prev_landmarks is not None:
#                 movement_detected = False
                
#                 for i in range(len(landmarks)):
#                     # Compute the Euclidean distance between corresponding landmarks
#                     distance = euclidean_distance(landmarks[i][0], prev_landmarks[i][0])
                    
#                     # If movement is detected, set the flag and break the loop
#                     if distance > MOVEMENT_THRESHOLD:
#                         movement_detected = True
#                         break
                
#                 # If movement is detected, print a message
#                 if movement_detected:
#                     print("Movement detected")
#                 else:
#                     print("Static frame detected")
            
#             # Update previous landmarks
#             prev_landmarks = landmarks.copy()

#         # Draw bounding boxes and landmarks on the frame
#         for bbox, landmark in zip(bboxes, landmarks):
#             x1, y1, x2, y2, _ = bbox
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
#             for point in landmark:
#                 cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

#         # Show the result in a window
#         cv2.imshow("Face Detection", frame)

#         # Press 'Q' on the keyboard to exit
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             break

#     # Release camera and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()











# import cv2
# import numpy as np
# from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face

# # Initialize the face detector
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5m-face.pt")
# # detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# # Define a function to compute the Euclidean distance between two points
# def euclidean_distance(pt1, pt2):
#     return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# # Define a threshold for movement detection
# MOVEMENT_THRESHOLD = 5  # Adjust as needed

# def main():
#     # Open the camera
#     cap = cv2.VideoCapture(0)

#     # Initialize variables
#     prev_eye_center = None

#     # Read frames from the camera
#     while True:
#         # Capture a frame from the camera
#         _, frame = cap.read()

#         # Get faces and landmarks using the face detector
#         bboxes, landmarks = detector.detect(image=frame)
        
#         # Check if any faces are detected
#         if len(landmarks) > 0:
#             # Extract landmarks for the first detected face
#             landmarks_face = landmarks[0]
            
#             # Extract the landmarks for the left and right eyes
#             left_eye = landmarks_face[36:42]
#             right_eye = landmarks_face[42:48]

#             # Calculate the center of each eye
#             left_eye_center = np.mean(left_eye, axis=0).astype(np.int)
#             right_eye_center = np.mean(right_eye, axis=0).astype(np.int)

#             # Check for movement between consecutive frames
#             if prev_eye_center is not None:
#                 # Compute the movement of each eye
#                 left_eye_movement = euclidean_distance(left_eye_center, prev_eye_center[0])
#                 right_eye_movement = euclidean_distance(right_eye_center, prev_eye_center[1])

#                 # Print the direction of eye movement
#                 if left_eye_movement > MOVEMENT_THRESHOLD:
#                     print("Left eye movement detected")
#                 elif right_eye_movement > MOVEMENT_THRESHOLD:
#                     print("Right eye movement detected")

#             # Update the previous eye centers
#             prev_eye_center = (left_eye_center, right_eye_center)

#         # Draw bounding boxes and landmarks on the frame
#         for bbox, landmark in zip(bboxes, landmarks):
#             x1, y1, x2, y2, _ = bbox
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
#             for point in landmark:
#                 cv2.circle(frame, tuple(point), 1, (0, 255, 0), -1)

#         # Show the result in a window
#         cv2.imshow("Face Detection", frame)

#         # Press 'Q' on the keyboard to exit
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             break

#     # Release camera and close all OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
