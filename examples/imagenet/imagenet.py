import numpy as np
import classes
import os
import gati
from PIL import Image

def preprocess(image):
    if not os.path.exists(image):
        raise OSError("File not found: {}".format(image))
    img = Image.open(image)
    # resize to (256,256)
    img = img.resize((256,256))
    img = np.array(img.convert('RGB'))
    # scale b/w 0 and 1
    img = img / 255.
    # take a (224,224) center crop of the image
    h, w = img.shape[0], img.shape[1]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0 : y0+224, x0 : x0+224, :]
    # Normalize (values obtained from millions of imagenet images)
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, axes=[2, 0, 1])
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def gen():
    arr = np.load("imagenet_100.npy")
    return arr

def post(arr):
    label = np.argmax(arr)
    with open("results.txt", "a") as f: f.write(f"{label}\n")
    print(f"{label} {classes.imagenet_labels[label]}")
    return label

if __name__ == "__main__":
    onnx_path = "../../models/vgg_16_224_int8.onnx"
    bitstream = "../../rah_gati_0.4.1.hex"
    gml_path = "model.gml"
    gati.set_arch(ramsize=512, sa_arch="9,4,4", vasize=32, accbuf_size=4096, fcbuf_size=32768)
    gati.compile(onnx_path, gml_path, "pretty-print-inst")
    gati.flash(bitstream)
    with open("results.txt", "w"): pass
    #gati.run(onnx_path, gml_path, "imagenet.py", "preprocess", "post", "inputpath ../../img.jpg")
    gati.run(onnx_path, gml_path, "imagenet.py", "gen", "post")
    print(f"Match: {gati.match('imagenet_100_labels.txt', 'results.txt')}%")
