import argparse
import os
import cv2
import shutil
import numpy as np
from PIL import Image
import gati

is_video = False
is_image_dir = False
is_single_image = False
temp_in_dir = None
dir_image = None


script_dir = os.path.dirname(os.path.abspath(__file__))

def preprocess(input_data, target_size=(300, 300)):
    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise OSError(f"File not found: {input_data}")
        img = Image.open(input_data).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img)
    elif isinstance(input_data, np.ndarray):
        img = cv2.resize(input_data, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError("input_data must be a file path or a numpy array")
    img = img.astype(np.float32) / 255.0
    img = (img - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.reshape(1, 1, 3, *target_size)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img


def reshape_scores(score_map, num_classes=21, num_anchors=6):
    n, c, h, w = score_map.shape 
    score_map = score_map.reshape(n, num_anchors, num_classes, h, w)
    score_map = score_map.transpose(0, 3, 4, 1, 2) 
    return score_map.reshape(n, -1, num_classes)    


def reshape_boxes(box_map, num_anchors=6):
    n, c, h, w = box_map.shape  
    box_map = box_map.reshape(n, num_anchors, 4, h, w)
    box_map = box_map.transpose(0, 3, 4, 1, 2)     
    return box_map.reshape(n, -1, 4)       

def decode_boxes(loc, priors, variances=[0.1, 0.2]):

    cx = priors[:, 0] + loc[:, 0] * variances[0] * priors[:, 2]
    cy = priors[:, 1] + loc[:, 1] * variances[0] * priors[:, 3]

    w = priors[:, 2] * np.exp(loc[:, 2] * variances[1])
    h = priors[:, 3] * np.exp(loc[:, 3] * variances[1])

    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    boxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
    return boxes 


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold=0.5, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    indexes = np.argsort(-scores)[:candidate_size]

    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current)
        if (0 < top_k == len(picked)) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]

        ious = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[ious <= iou_threshold]

    return box_scores[picked, :]

def nms(box_scores, score_threshold=0.4,
        iou_threshold=0.5, sigma=0.5, top_k=-1, candidate_size=200):

        return hard_nms(box_scores, iou_threshold, top_k, candidate_size)   

def postprocess(boxes, scores, width, height, class_names,
                prob_threshold=0.4, iou_threshold=0.5, top_k=200):

    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, scores.shape[1]):  
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)

        box_probs = nms(box_probs, prob_threshold,
                        iou_threshold=iou_threshold, top_k=top_k)

        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.zeros((0, 4)), np.array([]), np.array([])

    picked_box_probs = np.vstack(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return picked_box_probs[:, :4], np.array(picked_labels), picked_box_probs[:, 4]  


def run_inference(onnx_path, img, frame):
    output_tensor = gati.sim(onnx_path, {name: img})
    outputs = [out[1] for out in (output_tensor)]
    scores = outputs[1:12:2]
    boxes  = outputs[0:12:2]

    all_scores = [reshape_scores(s) for s in scores]
    all_boxes = [reshape_boxes(b) for b in boxes]

    final_scores = np.concatenate(all_scores, axis=1)
    final_boxes  = np.concatenate(all_boxes, axis=1)

    exp_a = np.exp(final_scores - np.max(final_scores, axis=-1, keepdims=True))
    softmax_a = exp_a / np.sum(exp_a, axis=-1, keepdims=True)
    final_scores = softmax_a

    final_boxes = final_boxes.squeeze()
    decoded_boxes = decode_boxes(final_boxes, priors)
    decoded_boxes = np.expand_dims(decoded_boxes, axis=0)

    h, w = frame.shape[:2]
    Scores = final_scores.squeeze()
    Boxes = decoded_boxes.squeeze()

    final_boxes, labels, probs = postprocess(Boxes, Scores, w, h, class_names)

    for i in range(final_boxes.shape[0]):
        box = final_boxes[i, :].astype(int)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(frame, label, (box[0] + 5, box[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        

    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX inference on NumPy data.")
    parser.add_argument("-i", "--image", help="Path to an image or directory of images.")
    parser.add_argument("-v", "--video", help="Path to a video file.")
    parser.add_argument("-c", "--camera", action="store_true", help="Use camera for detection.")
    args = parser.parse_args()
    onnx_path = "mobilenetv2_ssd_lite_tailfree_VOC_mAP_57.onnx"
    class_names = ["bg", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair",
                    "cow", "diningtable", "dog", "horse", "motorbike",
                    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    priors = np.load("priors.npy")
    name = gati.get_model_inputs(onnx_path)[0]


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
        cap = cv2.VideoCapture(0)  
        if not cap.isOpened():
            print("Could not open camera")
            exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = preprocess(frame)
            frame = run_inference(onnx_path, img, frame)
            cv2.imshow("Gati Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        parser.print_help()
        exit(1)

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
        orig_image = run_inference(onnx_path, img, cv2.imread(image_path))
        cv2.imwrite(f"output_{fname}" ,orig_image)
        

dir_image="/home/nirdesh/vicharak/sysim/examples/voc/"
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