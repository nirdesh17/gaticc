import numpy as np
import classes
import gati
import cv2
from PIL import Image


def preprocess(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img = img.resize((256, 256))
    img = np.array(img, dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    y0 = (h - 224) // 2
    x0 = (w - 224) // 2
    img = img[y0:y0+224, x0:x0+224, :]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

if __name__=="__main__":
  onnx_path="model.onnx"

  k = gati.get_model_inputs(onnx_path)[0]
  while True:
    cap = cv2.VideoCapture(0) # if using IP Webcam enter the URL instead of 0 inside the quotes
    if not cap.isOpened():
        print("Could not open camera")
        exit(1)
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Input', frame)
    a = preprocess(frame)
    ret=post(gati.sim(onnx_path,{k:a}))
    print(f"Output: {classes.imagenet_labels[ret[0]]}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
  cv2.destroyAllWindows()
  
