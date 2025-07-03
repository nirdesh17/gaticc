import gati
import numpy as np
import os

max_output_boxes = np.array([100], dtype=np.float32)  # cast to float
iou_threshold = np.array([0.44999998807907104], dtype=np.float32)
score_threshold = np.array([0.5], dtype=np.float32)

def nms_preprocess():
    bboxes = np.load("input_bboxes.npy") 
    scores = np.load("raw_input_class_scores.npy")
    scores = np.transpose(scores, (0, 2, 1))
    print(f'boxes shape : {bboxes.shape}')
    print(f'Scores shape: {scores.shape}')
    inputs = {
    "boxes": bboxes,
    "scores": scores,
    "max_output_boxes": max_output_boxes,
    "iou_threshold":iou_threshold,
    "score_threshold":score_threshold
    }
    return inputs

#dummy post process ! 
#WARNING this doesn't work !!!
def nms_postprocess():
    pass
    # Load ground truth and NMS results
    # ground_truth = np.load("ground_truth_detections.npy", allow_pickle=True)
    # nms_results = np.load("nms_results.npy", allow_pickle=True)
    # nms_results = np.load("outputs/nms_results.npy", allow_pickle=True)

    # Compare results for each image
    # for image_idx, (gt_detections, nms_detections) in enumerate(zip(ground_truth, nms_results)):
    #     print(f"Image {image_idx + 1}:")
    #     print("Ground Truth Detections:\n", gt_detections)
    #     print("NMS Detections:\n", nms_detections)
    #     print("-----------------------------")

if __name__ == "__main__":
    onnx_path = "../../models/nms_operator_new.onnx"
    bitstream = "../../rah.hex"
    gml_path = "model.gml"
    gati.summary(onnx_path)
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile(onnx_path, gml_path,"print-exec-graph","pretty-print-blob","pretty-print-inst","v")
    # os.system('hexyl model.gml')
    # gati.flash(bitstream)
    # with open("results.txt", "w"): pass
    gati.run(onnx_path, gml_path, nms_preprocess(),"verbose","verbose2","dry-run")
    # print(f"Match: {gati.match('cifar_100_labels.txt', 'results.txt')}%")