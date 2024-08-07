import onnxruntime

import numpy as np
import os
import cv2
import datetime

# type: ignore
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict

global label_dic
label_dic = {"drawings": 0, "hentai": 1, "neutral": 2, "porn": 3, "sexy": 4}


def get_key(dict, value):
    """
    return key given a value. From a dictionary
    """
    for key, val in dict.items():
        if val == value:
            return key
    return "Value not found"


def predict_video_frames(
    video_path: str,
    model=None,
    processor=None,
    progress_bar: bool = True,
    device: str = "cuda",
    output_video_path: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """
    Make prediction for each video frame.
    """
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    fps = cap.get(cv2.CAP_PROP_FPS)  # pylint: disable=no-member
    print(f" Frames per second: {fps}\n")
    nsfw_probabilities: List[float] = []
    frame_count = 0
    images = []
    if progress_bar:
        pbar = tqdm(
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )  # pylint: disable=no-member
    else:
        pbar = None

    while cap.isOpened():
        ret, bgr_frame = cap.read()  # Get next video frame.
        if not ret:
            break  # End of given video.

        if pbar is not None:
            pbar.update(1)

        frame_count += 1
        frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
        frame = cv2.resize(frame, (224, 224), cv2.INTER_AREA)
        img = np.array(frame, dtype=np.float32)
        img /= 255.0
        img = np.transpose(img, (2, 0, 1))

        images.append(img)

    print("\nEnd of Video. Inference of Frames")

    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member

    if pbar is not None:
        pbar.close()
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    outputs = model.run([output_name], {input_name: images})[0]
    logits = np.array(outputs)

    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    for probs in probabilities:
        output_probs = {}
        for prob, key in zip(probs, range(0, len(probs))):
            label = get_key(label_dic, key)
            output_probs[label] = prob

        # probability nsfw
        prob_nsfw = output_probs["sexy"] + output_probs["hentai"] + output_probs["porn"]
        nsfw_probabilities.append(prob_nsfw)

    return nsfw_probabilities


time1 = datetime.datetime.now()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ort_sess = onnxruntime.InferenceSession(
    "./models/onnx/vit_nsfw.onnx", providers=["CPUExecutionProvider"]
)


img_paths = [
    os.path.join(ROOT_DIR, "images", "production_id_39977981.mp4"),
]


nsfw_probabilities = predict_video_frames(model=ort_sess, video_path=img_paths[0])
time2 = datetime.datetime.now()
secs = time2 - time1
print(f"Probability Non-Safe for Work {np.mean(nsfw_probabilities)}. Time: {secs}")
