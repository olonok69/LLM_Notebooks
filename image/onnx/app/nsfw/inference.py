"""
Inference utilities.
"""

from typing import List, Optional, Tuple, Dict
from transformers import ViTImageProcessor, ViTForImageClassification
import cv2
import numpy as np
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore
import torch

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


def predict_image_pytorch(
    image_path: str,
    processor: ViTImageProcessor = None,
    model: ViTForImageClassification = None,
    device: str = "cuda",
) -> Dict:
    """
    Pipeline from single image path to predicted NSFW probability.
    Optionally generate and save the Grad-CAM plot.
    """
    image = Image.open(image_path)
    # Feature Transform with  ViTImageProcessor
    inputs = processor(images=image, return_tensors="pt")
    # data To device
    inputs = inputs.to(device)
    # get prediction
    outputs = model(**inputs)
    # get logits
    logits = outputs.logits
    # get probabilities
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
    probs = list(probs[0])
    output_probs = {}
    for prob, key in zip(probs, range(0, len(probs))):
        label = get_key(label_dic, key)
        output_probs[label] = float(prob)

    return output_probs


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
    print(fps)
    video_writer: Optional[cv2.VideoWriter] = None  # pylint: disable=no-member
    nsfw_probability = 0.0
    nsfw_probabilities: List[float] = []
    frame_count = 0

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

        if video_writer is None and output_video_path is not None:
            video_writer = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame.shape[1], frame.shape[0]),
            )

        inputs = processor(images=frame, return_tensors="pt")
        inputs = inputs.to(device)
        outputs = model(**inputs)
        logits = outputs.logits

        # get probabilities
        probs = torch.softmax(logits, dim=1).cpu().detach().numpy()
        probs = list(probs[0])
        output_probs = {}
        for prob, key in zip(probs, range(0, len(probs))):
            label = get_key(label_dic, key)
            output_probs[label] = prob
        # probability nsfw
        prob_nsfw = output_probs["sexy"] + output_probs["hentai"] + output_probs["porn"]
        nsfw_probabilities.append(prob_nsfw)

        if video_writer is not None:
            prob_str = str(np.round(nsfw_probability, 2))
            result_text = f"NSFW probability: {prob_str}"
            # RGB colour.
            colour = (255, 0, 0) if nsfw_probability >= 0.8 else (0, 0, 255)
            cv2.putText(  # pylint: disable=no-member
                frame,
                result_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                1,
                colour,
                2,
                cv2.LINE_AA,  # pylint: disable=no-member
            )
            video_writer.write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
            )

    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()  # pylint: disable=no-member

    if pbar is not None:
        pbar.close()

    elapsed_seconds = (np.arange(1, len(nsfw_probabilities) + 1) / fps).tolist()
    return elapsed_seconds, nsfw_probabilities
