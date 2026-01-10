# import onnxruntime as ort
import os
import numpy as np
# import sys
import cv2
# import json
# from tqdm import tqdm
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
import gati
import argparse

is_video = False
is_image_dir = False
is_single_image = False
temp_in_dir = None
dir_image = None


script_dir = os.path.dirname(os.path.abspath(__file__))

IOU_THRESHOLD = 0.45
OBJ_THRESH = 0.4
NMS_THRESH = 0.45
CLASSES = (
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
            "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)



def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,640), 
                         interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img

def dfl(position):
    # Assuming position is a numpy array
    x = np.array(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    
    # Apply softmax on the 2nd dimension (mc)
    y = np.exp(y) / np.exp(y).sum(axis=2, keepdims=True)
    
    acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
    y = (y * acc_metrix).sum(axis=2)
    
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    grid = np.stack((col, row), axis=0)
    stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    
    box_confidences = box_confidences.ravel()

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = class_max_score * box_confidences >= OBJ_THRESH
    
    scores = (class_max_score[_class_pos]* box_confidences[_class_pos])

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]

        return np.array(keep)

def draw(image, boxes, scores, classes, height, width):
        """Draw the boxes on the image.

        # Argument:
            image: original image.
            boxes: ndarray, boxes of objects.
            classes: ndarray, classes of objects.
            scores: ndarray, scores of objects.
            all_classes: all classes name.
        """
        
        for box, score, cl in zip(boxes, scores, classes):
            print(box, score, cl)   
            top, left, right, bottom = box
            top = int(top*width/640) # x1
            left = int(left*height/640) # y1
            right = int(right*width/640) # x2
            bottom = int(bottom*height/640) # y2
            #print(top, left, right, bottom)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f} {2}'.format(CLASSES[cl], score, cl),
                        (top, left - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 5)
        
        cv2.imwrite('output.jpg', image)
        # cv2.imshow('frame', image)
        # cv2.waitKey(0)

# 2014 to 2017
class_map = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, ]


    
    
def postprocess(inference_output, frame, height, width):
    boxes, classes_conf, scores = [], [], []
    default_branch = 3
    pair_per_branch = len(inference_output) // default_branch

    for i in range(default_branch):
        boxes.append(box_process(inference_output[pair_per_branch * i]))
        classes_conf.append(inference_output[pair_per_branch * i + 1])
        scores.append(np.ones_like(
            inference_output[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        return _in.transpose(0, 2, 3, 1).reshape(-1, _in.shape[1])

    boxes = np.concatenate([sp_flatten(b) for b in boxes])
    classes_conf = np.concatenate([sp_flatten(c) for c in classes_conf])
    scores = np.concatenate([sp_flatten(s) for s in scores])
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)


    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(np.full_like(keep, c))
            nscores.append(s[keep])


    if nclasses or nscores:
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        for box, cls, scr in zip(boxes, classes, scores):
          box = [box[0]*width/640, box[1]*height/640, box[2]*width/640, box[3]*height/640]
          box_ = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
          ss = class_map[int(cls)]
    draw(frame, boxes, scores, classes, height, width)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run ONNX inference on NumPy data.")
    parser.add_argument("-i", "--image", help="Path to an image or directory of images.")
    parser.add_argument("-v", "--video", help="Path to a video file.")
    parser.add_argument("-c", "--camera", action="store_true", help="Use camera for detection.")
    args = parser.parse_args()
    onnx_path = "/home/nirdesh/hdd/gaticc/onnx/yolov8n_quantized_nonms_mAP_20_3.onnx"
   

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
            # frame = run_inference(onnx_path, img, frame)
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
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        input = preprocess(image)
        outputs = gati.sim(onnx_path, {name: input})
        new_outputs = []
        for i in range(len(outputs)):
          new_outputs.append(outputs[i][1])
        outputs = new_outputs
        postprocess(outputs, image, height, width)



        

        
     

