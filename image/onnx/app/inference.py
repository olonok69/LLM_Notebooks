import onnxruntime
from PIL import Image
import numpy as np
import os
import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

time1 = datetime.datetime.now()
ort_sess = onnxruntime.InferenceSession(
    "./models/onnx/vit_nsfw.onnx", providers=["CPUExecutionProvider"]
)

input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name

img_paths = [
    os.path.join(ROOT_DIR, "images", "n1.jpg"),
    # os.path.join(ROOT_DIR, "images", "hen1.jpg"),
]

label_dic = {"drawings": 0, "hentai": 1, "neutral": 2, "porn": 3, "sexy": 4}


def get_key(dict, value):
    """
    return key given a value. From a dictionary
    """
    for key, val in dict.items():
        if val == value:
            return key
    return "Value not found"


def load_img(path):
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224), Image.BILINEAR)
    img = np.array(image, dtype=np.float32)
    img /= 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


inputs = np.array([load_img(path) for path in img_paths])
outputs = ort_sess.run([output_name], {input_name: inputs})[0]

logits = np.array(outputs)

probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

predicted_class = np.argmax(probabilities, axis=1)

print("Predicted classes :", predicted_class)
print("\n")
output_probs = {}
for prob, key in zip(probabilities[0], range(0, len(probabilities[0]))):
    label = get_key(label_dic, key)
    output_probs[label] = float(prob)
time2 = datetime.datetime.now()
print(output_probs)
secs = time2 - time1
print(f"Time: {secs}")
