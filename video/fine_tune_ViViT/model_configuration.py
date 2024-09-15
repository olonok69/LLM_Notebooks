import numpy as np
from datasets import load_metric
from transformers import  VivitConfig,VivitForVideoClassification
import torch
import evaluate

metric = evaluate.load("accuracy", trust_remote_code=True)
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([(torch.tensor(x['pixel_values']))  for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])}


def initialise_model(shuffled_dataset, device="cpu", model="google/vivit-b-16x2-kinetics400"):
    """initialize model
    """ 
    labels = shuffled_dataset['train'].features['labels'].names
    config = VivitConfig.from_pretrained(model)
    config.num_classes=len(labels)
    config.id2label = {str(i): c for i, c in enumerate(labels)}
    config.label2id = {c: str(i) for i, c in enumerate(labels)}
    config.num_frames=10
    config.video_size= [10, 224, 224]
    
    model = VivitForVideoClassification.from_pretrained(
    model,
    ignore_mismatched_sizes=True,
    config=config,).to(device)
    return model 