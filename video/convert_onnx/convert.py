### Environment  WSL 2 Linux Debian 
### Python 3.10.14
### packages in requirements.txt

from data_handling import frames_convert_and_create_dataset_dictionary
import os
import numpy as np
import av
from pathlib import Path
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import torch
from IPython import embed
from transformers.utils import logging
logging.set_verbosity_error() 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

from onnxruntime.tools import pytorch_export_contrib_ops
pytorch_export_contrib_ops.register()


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            reformatted_frame = frame.reformat(width=224,height=224)
            frames.append(reformatted_frame)
    new=np.stack([x.to_ndarray(format="rgb24") for x in frames])

    return new


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Path Fine Tuned Model
artifact_dir = "/mnt/d/repos2/video/artifacts/ViViT-Fine-tuned:v0"

# Configure Fine Tuned Model
labels = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress']
config = VivitConfig.from_pretrained(artifact_dir)
config.num_classes=len(labels)
config.id2label = {str(i): c for i, c in enumerate(labels)}
config.label2id = {c: str(i) for i, c in enumerate(labels)}
config.num_frames=10
config.video_size= [10, 224, 224]

# Load Image Processor and Fine Tuned Model
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
fine_tune_model = VivitForVideoClassification.from_pretrained(artifact_dir,config=config)

# Process Video example
file_name = "./tmp/6540601-uhd_2560_1440_25fps.mp4"
container = av.open(file_name)
indices = sample_frame_indices(clip_len=10, frame_sample_rate=1,seg_len=container.streams.video[0].frames)
print(f"Processing file {file_name} number of Frames: {container.streams.video[0].frames}")  
video = read_video_pyav(container=container, indices=indices)
inputs = image_processor(list(video), return_tensors="pt")

# You can choose between take the image or generate a randon value. It is just an input
input_tensor = inputs["pixel_values"]
input_tensor =  torch.randn(1,  10, 3, 224, 224)

torch.onnx.export(
    model=fine_tune_model,
    args=(input_tensor,),
    f="onnx/vivit.onnx",
    opset_version=15,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)