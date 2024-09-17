import datetime
import os
import numpy as np
import av
import onnxruntime
from IPython import embed
from transformers import VivitImageProcessor
from transformers.utils import logging
logging.set_verbosity_error() 
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings

# Disable a few less-than-useful UserWarnings from setuptools and pydantic
warnings.filterwarnings("ignore", category=UserWarning)

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

def get_key(dict, value):
    """
    return key given a value. From a dictionary
    """
    for key, val in dict.items():
        if val == value:
            return key
    return "Value not found"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

time1 = datetime.datetime.now()
# providers = [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
#                                         "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]
ort_sess = onnxruntime.InferenceSession(
    "./onnx/vivit.onnx", providers=["CPUExecutionProvider"]
)
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

input_name = ort_sess.get_inputs()[0].name
output_name = ort_sess.get_outputs()[0].name


label_dic = {'ApplyEyeMakeup':0, 'ApplyLipstick':1, 'Archery':2, 'BabyCrawling':3, 'BalanceBeam':4, 'BandMarching':5, 
             'BaseballPitch':6, 'Basketball':7,'BasketballDunk':8, 'BenchPress':9}

file_name = "./tmp/6540601-uhd_2560_1440_25fps.mp4"
container = av.open(file_name)
indices = sample_frame_indices(clip_len=10, frame_sample_rate=1,seg_len=container.streams.video[0].frames)
print(f"Processing file {file_name} number of Frames: {container.streams.video[0].frames}")  
video = read_video_pyav(container=container, indices=indices)

#Process Video
inputs_t = np.array(image_processor(list(video), return_tensors="pt")['pixel_values'])
#inputs = np.array([video.astype(np.float32)])['pixel_values']
# Inference IN CPU
time1b = datetime.datetime.now()
outputs = ort_sess.run([output_name], {input_name: inputs_t})[0]
# Get Logits
logits = np.array(outputs)
# Get Probabilities
probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
# Get Pedicted Class
predicted_class = np.argmax(probabilities, axis=1)

print(f"Predicted classes: {predicted_class[0]}, label: { get_key(label_dic, predicted_class[0])}")
print("\n")
output_probs = {}
print("All Probabilities:")
for prob, key in zip(probabilities[0], range(0, len(probabilities[0]))):
    label = get_key(label_dic, key)
    output_probs[label] = float(prob)
time2 = datetime.datetime.now()
print(output_probs)
secs = time2 - time1
secsb = time2 - time1b
print(f"Time total Inferencia: {secsb}")
print(f"Time total: {secs}")