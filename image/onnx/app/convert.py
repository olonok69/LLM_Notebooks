import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import os
import sys
import platform
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if platform.system() == "Windows":
    MODELS_PATH = os.path.join(ROOT_DIR, "models")


print(MODELS_PATH)
try:
    model_path = os.path.join(MODELS_PATH, "nsfw_pytorch")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model = model.to(device)
except Exception as e:
    ex_type, ex_value, ex_traceback = sys.exc_info()
    print(f"Exception {ex_type} value {str(ex_value)}")


image_path = "./images/n1.jpg"
image = Image.open(image_path).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

input_tensor = inputs["pixel_values"]


torch.onnx.export(
    model=model,
    args=(input_tensor,),
    f="models/onnx/vit_nsfw.onnx",
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
