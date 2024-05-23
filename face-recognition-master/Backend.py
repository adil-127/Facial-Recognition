
# import asyncio
# import uvicorn
# import threading
# import time
# import cv2
# import numpy as np
# import torch
# import yaml
# from torchvision import transforms
# from queue import Queue
# from fastapi import FastAPI
# from pydantic import BaseModel
# import base64

# from face_alignment.alignment import norm_crop
# from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face
# from face_recog.arcface.model import iresnet_inference
# from face_recog.arcface.utils import compare_encodings, read_features
# from face_tracking.tracker.byte_tracker import BYTETracker
# from face_tracking.tracker.visualize import plot_tracking
# from Anti_spoofing.test import test

# # Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Face detector (choose one)
# detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# # detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

# recognizer = iresnet_inference(
#     model_name="r100", path="face_recog/arcface/weights/arcface_r100.pth", device=device
# )

# images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# id_face_mapping = {}

# data_mapping = {
#     "raw_image": [],
#     "tracking_ids": [],
#     "detection_bboxes": [],
#     "detection_landmarks": [],
#     "tracking_bboxes": [],
# }

# frame_queue = Queue()

# app = FastAPI()

# class FrameData(BaseModel):
#     frame: str

# @app.post("/process-frame/")
# async def Getframe(frame_data: FrameData):
#     frame_bytes = base64.b64decode(frame_data.frame)
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     frame_queue.put(frame)
#     main()

#     return {"status": "frame received"}


# @app.get("/latest-caption/")
# async def get_latest_caption():
#     if not data_mapping["tracking_ids"]:
#         return {'latest_caption':'No caption available'}
    
#     latest_tracking_id = data_mapping["tracking_ids"][-1]
#     caption = id_face_mapping.get(latest_tracking_id, "No caption available")
#     return {"latest_caption": caption}



# def load_config(file_name):
#     with open(file_name, "r") as stream:
#         try:
#             return yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)

# def process_tracking(frame, detector, tracker, args, frame_id, fps):
#     outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
#     tracking_tlwhs = []
#     tracking_ids = []
#     tracking_scores = []
#     tracking_bboxes = []

#     if outputs is not None:
#         online_targets = tracker.update(
#             outputs, [img_info["height"], img_info["width"]], (128, 128)
#         )
#         for i in range(len(online_targets)):
#             t = online_targets[i]
#             tlwh = t.tlwh
#             tid = t.track_id
#             vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
#             if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
#                 x1, y1, w, h = tlwh
#                 tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
#                 tracking_tlwhs.append(tlwh)
#                 tracking_ids.append(tid)
#                 tracking_scores.append(t.score)

#         tracking_image = plot_tracking(
#             img_info["raw_img"],
#             tracking_tlwhs,
#             tracking_ids,
#             names=id_face_mapping,
#             frame_id=frame_id + 1,
#             fps=fps,
#         )
#     else:
#         tracking_image = img_info["raw_img"]

#     data_mapping["raw_image"] = img_info["raw_img"]
#     data_mapping["detection_bboxes"] = bboxes
#     data_mapping["detection_landmarks"] = landmarks
#     data_mapping["tracking_ids"] = tracking_ids
#     data_mapping["tracking_bboxes"] = tracking_bboxes

#     return tracking_image

# @torch.no_grad()
# def get_feature(face_image):
#     face_preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((112, 112)),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#     ])
#     face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#     face_image = face_preprocess(face_image).unsqueeze(0).to(device)
#     emb_img_face = recognizer(face_image).cpu().numpy()
#     images_emb = emb_img_face / np.linalg.norm(emb_img_face)
#     return images_emb

# def recognition(face_image):
#     query_emb = get_feature(face_image)
#     score, id_min = compare_encodings(query_emb, images_embs)
#     name = images_names[id_min]
#     score = score[0]
#     return score, name

# def mapping_bbox(box1, box2):
#     x_min_inter = max(box1[0], box2[0])
#     y_min_inter = max(box1[1], box2[1])
#     x_max_inter = min(box1[2], box2[2])
#     y_max_inter = min(box1[3], box2[3])
#     intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
#     area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
#     area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
#     union_area = area_box1 + area_box2 - intersection_area
#     iou = intersection_area / union_area
#     return iou

# async def tracking(detector, args):
#     start_time = time.time_ns()
#     frame_count = 0
#     fps = -1
#     tracker = BYTETracker(args=args, frame_rate=30)
#     frame_id = 0

#     while True:
#         frame = frame_queue.get()
#         tracking_image = process_tracking(frame, detector, tracker, args, frame_id, fps)
#         frame_count += 1
#         if frame_count >= 30:
#             fps = 1e9 * frame_count / (time.time_ns() - start_time)
#             frame_count = 0
#             start_time = time.time_ns()
#         # cv2.imshow("Face Recognition", tracking_image)
#         # ch = cv2.waitKey(1)
#         ch=''
#         if ch == 27 or ch == ord("q") or ch == ord("Q"):
#             break

# def recognize():
#     while True:
#         raw_image = data_mapping["raw_image"]
#         detection_landmarks = data_mapping["detection_landmarks"]
#         detection_bboxes = data_mapping["detection_bboxes"]
#         tracking_ids = data_mapping["tracking_ids"]
#         tracking_bboxes = data_mapping["tracking_bboxes"]

#         for i in range(len(tracking_bboxes)):
#             for j in range(len(detection_bboxes)):
#                 mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
#                 if mapping_score > 0.9:
#                     face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])
#                     label, value, speed = test(
#                         image=raw_image,
#                         model_dir='/media/adilamir/New Volume/wenava/face-attendance-system/Anti_spoofing/resources/anti_spoof_models',
#                         device_id=0
#                     )
#                     if label == 1:
#                         score, name = recognition(face_image=face_alignment)
#                         if name is not None:
#                             if score < 0.25:
#                                 caption = "UN_KNOWN"
#                             else:
#                                 caption = f"{name}:{score:.2f}"
#                     else:
#                         caption = 'Fake'
#                     print(caption)
#                     id_face_mapping[tracking_ids[i]] = caption
#                     detection_bboxes = np.delete(detection_bboxes, j, axis=0)
#                     detection_landmarks = np.delete(detection_landmarks, j, axis=0)
#                     break
#         if tracking_bboxes == []:
#             print("Waiting for a person...")




# def main():
#     file_name = "./face_tracking/config/config_tracking.yaml"
#     config_tracking = load_config(file_name)

#     # Start tracking thread
#     thread_track = threading.Thread(
#         target=lambda: asyncio.run(tracking(detector, config_tracking))
#     )
#     thread_track.start()

#     # Start recognition thread
#     thread_recognize = threading.Thread(target=recognize)
#     thread_recognize.start()

# if __name__ == "__main__":
    
#     asyncio.run(main())
#     uvicorn.run(app, host="0.0.0.0", port=8000)










import asyncio
import uvicorn
import threading
import time
import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms
from queue import Queue
from fastapi import FastAPI
from pydantic import BaseModel
import base64

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_detection.yolov5_face.detector import Yolov5Face
from face_recog.arcface.model import iresnet_inference
from face_recog.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
from Anti_spoofing.test import test
from collections import Counter


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector (choose one)
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")

recognizer = iresnet_inference(
    model_name="r100", path="face_recog/arcface/weights/arcface_r100.pth", device=device
)

images_names, images_embs = read_features(feature_path="./datasets/face_features/feature")

# id_face_mapping = {}

# data_mapping = {
#     "raw_image": [],
#     "tracking_ids": [],
#     "detection_bboxes": [],
#     "detection_landmarks": [],
#     "tracking_bboxes": [],
# }

frame_queue = Queue()

app = FastAPI()

class FrameData(BaseModel):
    frame: list

@app.post("/process-frame/")
async def Getframe(frame_data: FrameData):
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)
    caption_list=[]
    # for i in range (len(frame_data)):
    #     frame_bytes = base64.b64decode(frame_data.frame[i])
    #     nparr = np.frombuffer(frame_bytes, np.uint8)
    #     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #     caption=await tracking(detector, config_tracking,frame)
    #     caption_list.append(caption)
    # occurence_count = Counter(caption_list)    
    # caption=occurence_count.most_common(1)[0][0]
   
    for frame_bytes in frame_data.frame:  
        
        try:
            nparr = np.frombuffer(base64.b64decode(frame_bytes), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            caption = await tracking(detector, config_tracking, frame)
        except (ValueError) as e:  # Handle specific exceptions
            print(f"Error processing frame: {e}")
            caption = "Unknown"  # Default caption for recoverable errors
        caption_list.append(caption)

    occurence_count = Counter(caption_list)    
    caption=occurence_count.most_common(1)[0][0]    
        
    
    return {"status": "frame received",'classification':caption}


# @app.get("/latest-caption/")
# async def get_latest_caption():
#     if not data_mapping["tracking_ids"]:
#         return {'latest_caption':'No caption available'}
    
#     latest_tracking_id = data_mapping["tracking_ids"][-1]
#     caption = id_face_mapping.get(latest_tracking_id, "No caption available")
#     return {"latest_caption": caption}



def load_config(file_name):
    with open(file_name, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

async def process_tracking(frame, detector, tracker, args, frame_id, fps):
    data_mapping = {
    "raw_image": [],
    "tracking_ids": [],
    "detection_bboxes": [],
    "detection_landmarks": [],
    "tracking_bboxes": [],
}
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
    tracking_tlwhs = []
    tracking_ids = []
    tracking_scores = []
    tracking_bboxes = []

    if outputs is not None:
        online_targets = tracker.update(
            outputs, [img_info["height"], img_info["width"]], (128, 128)
        )
        for i in range(len(online_targets)):
            t = online_targets[i]
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args["aspect_ratio_thresh"]
            if tlwh[2] * tlwh[3] > args["min_box_area"] and not vertical:
                x1, y1, w, h = tlwh
                tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                tracking_tlwhs.append(tlwh)
                tracking_ids.append(tid)
                tracking_scores.append(t.score)

        tracking_image = plot_tracking(
            img_info["raw_img"],
            tracking_tlwhs,
            tracking_ids,
            # names=id_face_mapping,
            frame_id=frame_id + 1,
            fps=fps,
        )
    else:
        tracking_image = img_info["raw_img"]

    data_mapping["raw_image"] = img_info["raw_img"]
    data_mapping["detection_bboxes"] = bboxes
    data_mapping["detection_landmarks"] = landmarks
    data_mapping["tracking_ids"] = tracking_ids
    data_mapping["tracking_bboxes"] = tracking_bboxes
    print(data_mapping)
    caption= recognize(data_mapping=data_mapping)
    print(f"processT{caption}")

    return caption

@torch.no_grad()
def get_feature(face_image):
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    emb_img_face = recognizer(face_image).cpu().numpy()
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb

def recognition(face_image):
    query_emb = get_feature(face_image)
    score, id_min = compare_encodings(query_emb, images_embs)
    name = images_names[id_min]
    score = score[0]
    return score, name

def mapping_bbox(box1, box2):
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])
    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(0, y_max_inter - y_min_inter + 1)
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou

async def tracking(detector, args,frame):
    start_time = time.time_ns()
    frame_count = 0
    fps = -1
    tracker = BYTETracker(args=args, frame_rate=30)
    frame_id = 0

    # while True:
    frame = frame
    caption= await process_tracking(frame, detector, tracker, args, frame_id, fps)
        # frame_count += 1
        # if frame_count >= 30:
        #     fps = 1e9 * frame_count / (time.time_ns() - start_time)
        #     frame_count = 0
        #     start_time = time.time_ns()
        # cv2.imshow("Face Recognition", tracking_image)
        # ch = cv2.waitKey(1)
        # ch=''
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     # break
    return caption

def recognize(data_mapping):
    while True:
        raw_image = data_mapping["raw_image"]
        detection_landmarks = data_mapping["detection_landmarks"]
        detection_bboxes = data_mapping["detection_bboxes"]
        tracking_ids = data_mapping["tracking_ids"]
        tracking_bboxes = data_mapping["tracking_bboxes"]

        for i in range(len(tracking_bboxes)):
            for j in range(len(detection_bboxes)):
                mapping_score = mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                if mapping_score > 0.9:
                    face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])
                    label, value, speed = test(
                        image=raw_image,
                        model_dir='/media/adilamir/New Volume/wenava/face-attendance-system/Anti_spoofing/resources/anti_spoof_models',
                        device_id=0
                    )
                    if label == 1:
                        score, name = recognition(face_image=face_alignment)
                        if name is not None:
                            if score < 0.25:
                                caption = "UN_KNOWN"
                            else:
                                caption = f"{name}:{score:.2f}"
                    else:
                        caption = 'Fake'
                    print(caption)
                    # id_face_mapping[tracking_ids[i]] = caption
                    detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                    detection_landmarks = np.delete(detection_landmarks, j, axis=0)
                    break
        if tracking_bboxes == []:
            print("Waiting for a person...")
            caption='No_person'
            
        return caption



def main():
    file_name = "./face_tracking/config/config_tracking.yaml"
    config_tracking = load_config(file_name)

    # Start tracking thread
    thread_track = threading.Thread(
        target=lambda: asyncio.run(tracking(detector, config_tracking))
    )
    thread_track.start()

    # Start recognition thread
    thread_recognize = threading.Thread(target=recognize)
    thread_recognize.start()

if __name__ == "__main__":
    
    asyncio.run(main())
    uvicorn.run(app, host="0.0.0.0", port=8000)


