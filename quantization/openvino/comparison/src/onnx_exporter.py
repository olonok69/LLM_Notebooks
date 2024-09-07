import os
import torch
from torch.onnx import export, TrainingMode


class ONNXExporter:
    def __init__(self, model, device, onnx_path: str):
        self.model = model
        self.onnx_path = onnx_path
        self.device = device

    def export_model(self):
        self.model.eval()

        # Define dummy input tensor
        x = torch.randn(1, 3, 224, 224).to(self.device)

        if not os.path.exists(self.onnx_path):
            os.makedirs("models", exist_ok=True)

        # Export model as ONNX
        export(
            self.model,
            x,
            self.onnx_path,
            training=TrainingMode.EVAL,
            verbose=True,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"Model exported to {self.onnx_path}")
