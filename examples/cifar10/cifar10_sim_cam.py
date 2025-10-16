import numpy as np
import gati
import cv2
from PIL import Image

cifar10_labels = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def post(arr):
  m = np.argmax(np.squeeze(np.stack([i[1] for i in arr]), axis=1), axis=-1)
  return m

def preprocess(image):
   img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
   img = img.resize((32, 32))
   img_array = np.array(img).astype(np.float32) / 255.0
   mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
   std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
   img_array = (img_array - mean) / std
   img_array = np.transpose(img_array, (2, 0, 1))
   img = np.expand_dims(img, axis=0)
   img = np.expand_dims(img, axis=0)
   img = np.ascontiguousarray(img, dtype=np.float32)
   return img

if __name__ == "__main__":
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
    print(f"Output: {cifar10_labels[ret[0]]}")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap.release()
  cv2.destroyAllWindows()
  