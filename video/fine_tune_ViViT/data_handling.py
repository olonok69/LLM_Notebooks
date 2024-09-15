import os
import numpy as np
import av
from pathlib import Path


def generate_all_files(root: Path, only_files: bool = True):
    for p in root.rglob("*"):
        if only_files and not p.is_file():
            continue
        yield p
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


def frames_convert_and_create_dataset_dictionary(directory):
    class_labels = []
    all_videos=[]
    video_files= []
    sizes = []
    for p in generate_all_files(Path(directory), only_files=True):
        set_files = str(p).split("/")[2] # train or test
        cls = str(p).split("/")[3] # class
        file= str(p).split("/")[4] # file name
        #file name path
        file_name= os.path.join(directory, set_files, cls, file)
        # print(f"Processing file {file_name}")    
        # Process class
        if cls not in class_labels:
            class_labels.append(cls)
        # process video File
        container = av.open(file_name)
        print(f"Processing file {file_name} number of Frames: {container.streams.video[0].frames}")  
        indices = sample_frame_indices(clip_len=10, frame_sample_rate=1,seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container=container, indices=indices)
        all_videos.append({'video': video, 'labels': cls})
        sizes.append(container.streams.video[0].frames)
    sizes = np.array(sizes)
    print(f"Min number frames {sizes.min()}")
    return all_videos, class_labels