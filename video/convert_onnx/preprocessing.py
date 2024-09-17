import numpy as np
import os
import av
import torch
from transformers import VivitImageProcessor, VivitModel
from data_handling import frames_convert_and_create_dataset_dictionary
from datasets import Dataset

image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
def process_example(example):
    inputs = image_processor(list(np.array(example['video'])), return_tensors='pt')
    inputs['labels'] = example['labels']
    return inputs



def create_dataset(video_dictionary):
    dataset = Dataset.from_list(video_dictionary)
    dataset = dataset.class_encode_column("labels")
    processed_dataset = dataset.map(process_example, batched=False)
    processed_dataset=processed_dataset.remove_columns(['video'])
    shuffled_dataset= processed_dataset.shuffle(seed=42)
    shuffled_dataset= shuffled_dataset.map(lambda x: {'pixel_values': torch.tensor(x['pixel_values']).squeeze()})
    shuffled_dataset =shuffled_dataset.train_test_split(test_size=0.1)
    return shuffled_dataset