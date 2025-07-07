import argparse
import os
import cv2
import shutil
import numpy as np
from PIL import Image
import gati


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to an image or directory of images.")
parser.add_argument("-v", "--video", help="Path to a video file.")
parser.add_argument("-c", "--camera", action="store_true", help="Use camera for detection.")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

is_video = False
is_image_dir = False
is_single_image = False
temp_in_dir = None
dir_image = None

if args.image:
    if os.path.isdir(args.image):  
        dir_image = args.image
        is_image_dir = True
    else:
        is_single_image = True
        dir_image = os.path.dirname(os.path.abspath(args.image))
        image_files = [os.path.basename(args.image)]  
elif args.video:
    is_video = True
    temp_in_dir = os.path.join(script_dir, "temp_in")  
    os.makedirs(temp_in_dir, exist_ok=True)

    dir_image = temp_in_dir
    cap = cv2.VideoCapture(args.video)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_frame = os.path.join(temp_in_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_frame, frame)
        idx += 1
    cap.release()
elif args.camera:
    is_video = True
    temp_in_dir = os.path.join(script_dir, "temp_in") 
    os.makedirs(temp_in_dir, exist_ok=True)

    dir_image = temp_in_dir
    cap = cv2.VideoCapture(0)
    idx = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_frame = os.path.join(temp_in_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_frame, frame)
        idx += 1
    cap.release()
else:
    parser.print_help()
    exit(1)

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((416, 416))
    img = np.array(img)
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def IOU(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x3, y3, x4, y4 = box2[:4]

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, x4 - x3) * max(0, y4 - y3)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def nms(boxes, iou_threshold=0.5):
    final_boxes = []
    classes = set(box[5] for box in boxes)
    for c in classes:
        class_boxes = [box for box in boxes if box[5] == c]
        class_boxes.sort(key=lambda x: x[4], reverse=True)
        while class_boxes:
            best_box = class_boxes.pop(0)
            final_boxes.append(best_box)
            class_boxes = [box for box in class_boxes if IOU(best_box, box) < iou_threshold]
    return final_boxes

anchors = [
    (1.08, 1.19),    
    (3.42, 4.41),    
    (6.63, 11.38),   
    (9.42, 5.11),   
    (16.62, 10.52)   
]

class_names = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

if is_single_image:
    files_to_process = image_files
elif is_image_dir:
    files_to_process = sorted([
        f for f in os.listdir(dir_image) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
elif is_video:
    files_to_process = sorted([
        f for f in os.listdir(dir_image) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

for idx, fname in enumerate(files_to_process):
    image_path = os.path.join(dir_image, fname)
    print(f"Processing {idx + 1}/{len(files_to_process)}: {image_path}")

    img = preprocess(image_path)
    img = np.ascontiguousarray(img)
    onnx_path = "/home/nirdesh/hdd/gaticc/onnx/tinyyolov2_quantized_mAP_44_VOC_PASCAL.onnx"

    output_tensor = gati.sim(onnx_path, img)
    output_name, raw_output = output_tensor[0]
    output_tensor = np.array(raw_output, dtype=np.float32)
    output_tensor = output_tensor[0]
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    output_tensor = output_tensor.reshape((13, 13, 5, 25))

    boxes = []
    for i in range(13):
        for j in range(13):
            for k in range(5):
                tx, ty, tw, th, to = output_tensor[i][j][k][:5]
                bx = (sigmoid(tx) + j) / 13
                by = (sigmoid(ty) + i) / 13
                bw = anchors[k][0] * np.exp(tw) / 13
                bh = anchors[k][1] * np.exp(th) / 13

                x1 = (bx - bw / 2) * 416
                y1 = (by - bh / 2) * 416
                x2 = (bx + bw / 2) * 416
                y2 = (by + bh / 2) * 416

                objectness_score = sigmoid(to)

                class_probs = output_tensor[i][j][k][5:25]
                class_probs = softmax(class_probs)
                c = np.argmax(class_probs)
                confidence_score = objectness_score * class_probs[c]

                if confidence_score > 0.4:
                    box = [x1, y1, x2, y2, confidence_score, c]
                    boxes.append(box)

    boxes = nms(boxes, iou_threshold=0.5)

    image = cv2.imread(image_path)
    image = cv2.resize(image, (416, 416))
    for box in boxes:
        x1, y1, x2, y2, confidence_score, class_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{class_names[class_id]}: {confidence_score:.2f}"
        print(f"Detected {label} at ({x1}, {y1}), ({x2}, {y2})")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    output_fname = f"output_{fname}"
    output_path = os.path.join(dir_image, output_fname)
    cv2.imwrite(output_path, image)


if is_video:
    output_files = sorted([
        f for f in os.listdir(dir_image) if f.startswith("output_") and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if output_files:
        first_frame = cv2.imread(os.path.join(dir_image, output_files[0]))
        h, w, _ = first_frame.shape
        out_video = os.path.join(script_dir, "final_output.mp4")  
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(out_video, fourcc, 30.0, (w, h))
        for out_file in output_files:
            frame = cv2.imread(os.path.join(dir_image, out_file))
            video_writer.write(frame)
        video_writer.release()
        print(f"Final video saved at: {out_video}")

    if temp_in_dir is not None and os.path.exists(temp_in_dir):
        shutil.rmtree(temp_in_dir)
        print(f"Temporary directory '{temp_in_dir}' deleted.")