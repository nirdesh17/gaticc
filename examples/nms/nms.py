import gati
import numpy as np

def nms_preprocess(onnx_path):
    inps = gati.get_model_inputs(onnx_path)
    boxes = np.load("input_bboxes.npy")
    scores = np.transpose(np.load("raw_input_class_scores.npy"), (0, 1, 3, 2))
    return {inps[0]: boxes, inps[1]: scores}

def nms_postprocess(out):
    print(out)

if __name__ == "__main__":
    onnx = "../../models/nms_operator_new.onnx"
    bitstream = "../../rah.hex"
    gml = "model.gml"
    gati.summary(onnx)
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile(onnx, gml, "print-exec-graph", "pretty-print-blob", "pretty-print-inst", "v")
    # gati.flash(bitstream)
    # out = gati.run(onnx, gml, nms_preprocess(onnx), "verbose", "verbose2", "dry-run")
    # print(f"Match: {gati.match('cifar_100_labels.txt', 'results.txt')}%")  # Optional matching for accuracy check
