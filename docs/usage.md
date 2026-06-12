Usage Guide
===========

This section explains how to use **GATICC** for compiling ONNX models,
running inference on the FPGA, and performing CPU-based simulation.

GATICC works in three simple stages:

1. **Preprocess your input** → produce a NumPy tensor  
2. **Run inference** (simulation or FPGA)  
3. **Postprocess the output** → get predictions, boxes, labels, etc.

GATICC does *not* perform preprocessing or postprocessing automatically.  
Users must implement these parts depending on their model (classification or detection).


Basic Workflow
--------------

A typical workflow looks like:

    ONNX Model  →  Preprocess Input (User code)
                 →  GATICC Compile (int8 only for FPGA)
                 →  Flash Bitstream
                 →  GATICC Load & Run
                 →  Postprocess Output (User code)



Preprocessing (User Responsibility)
--------------------------------------

GATICC expects input as a:

    {input_name: numpy_tensor}

Where:

- ``input_name`` comes from the ONNX model (use ``gati.get_model_inputs()``)
- ``numpy_tensor`` must match the model’s expected shape and dtype
- **GATICC expects input in *5D tensor format*: ``(N, 1, C, H, W)``**  
  This extra dimension is required by the internal architecture of GATI’s dataflow.

**Example: Classification Preprocess**

    import numpy as np
    import cv2

    def preprocess(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))       # CHW
        img = np.expand_dims(img, axis=0)        # NCHW
        return img


**Example: Object Detection Preprocess**

    def preprocess(img):
        img = cv2.resize(img, (300, 300))
        img = img[..., ::-1] / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        img = img.reshape(1, 1, 3, 300, 300)
        return img.astype(np.float32)


GATICC does **not** impose a fixed preprocessing style —only the final tensor
shape must follow ``(N, 1, C, H, W)``.  
Users may normalize, resize, or format inputs however their ONNX model requires.



Compiling INT8 ONNX Model for FPGA
-------------------------------------

FPGA inference requires **INT8 quantized ONNX models**.

    import gati

    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile("model_int8.onnx", "model.gml","plus more args if needed")


The compiler produces a ``.gml`` file used by the FPGA runtime.



Flash Bitstream (FPGA Only)
------------------------------

    gati.flash("gati_944.hex")



Load Model and Run Inference
-------------------------------


    name = gati.get_model_inputs("model_int8.onnx")[0]
    gati.load("model_int8.onnx", "model.gml")

    inp = preprocess("image.jpg")
    out = gati.run({name: inp})


The output is always a list of tuples:

    [("layer_name", numpy_array), ("layer_name", numpy_array), ...]

Users must apply postprocessing to convert raw tensors into:

- class IDs (classification)
- bounding boxes (object detection)
- scores / probabilities
- segmentation masks
- etc.



Simulation Mode (FP32/INT8)
------------------------------

Simulation runs on **CPU only**, no FPGA required.

Simulation supports:

- FP32 ONNX models  
- INT8 ONNX models  

    out = gati.sim("model_fp32.onnx", {name: inp})

Use this for debugging, comparison, and output verification.



Example: Classification Usage
--------------------------------

    import numpy as np
    import gati

    def post(arr):
        logits = np.stack([x[1] for x in arr])
        return np.argmax(logits, axis=-1)

    name = gati.get_model_inputs("mnist.onnx")[0]
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile("mnist_int8.onnx", "mnist.gml")
    gati.flash("gati_944.hex")
    gati.load("mnist_int8.onnx", "mnist.gml")

    inp = np.load("sample.npy")
    pred = post(gati.run({name: inp}))

    print("Prediction:", pred)



Example: Object Detection Usage (Simplified)
-----------------------------------------------


    img = preprocess("dog.jpg")
    name = gati.get_model_inputs("ssd.onnx")[0]

    outputs = gati.run({name: img})

    # User-written decode + NMS here
    boxes, labels, scores = decode(outputs)


(Full examples are included in the ``examples/`` folder.)



Important Notes
---------------

- GATICC does **not** do preprocessing or postprocessing.  
  Users must handle image normalization, resizing, decoding, NMS, etc.

- FPGA supports **only INT8** operators currently:
  
  - QLinearConv (normal / pointwise / depthwise)
  - Relu, Clip
  - QGemm
  - Flatten
  - QLinearAdd/Sub/Mul
  - MaxPool, AveragePool
  - QLinearSigmoid
  - QLinearConcat
  - QLinearLeakyRelu
  - Tanh
  - Split

- Simulation supports **FP32 and INT8**.

- Input/Output format depends entirely on the model architecture.



Example Files
-------------

The repository includes ready-to-run examples:

- ``examples/classification_run.py``
- ``examples/classification_sim.py``
- ``examples/detection_sim.py``
- ``examples/detection_run.py``

Users can copy these templates and plug in their own preprocessing and postprocessing code.

>[!NOTE]
>```
> Make sure to update the required paths in example files

More Usage & Full Command Reference
---------------------------------------

This guide covers the common workflow for compiling and running models.  
However, **GATICC provides many more runtime options, flags, and utilities**  
that users may need.

To view the full function reference, run:

    gaticc --help

This will show detailed documentation for:

- all available compiler flags  
- runtime options  
- model inspection commands  
- summary/info mode  
- architecture selection  
- dispatch options  
- debug/verbose options  
- explanations for each function


