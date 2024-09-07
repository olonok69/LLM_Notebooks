import os
import logging
import onnxruntime as ort
import numpy as np
from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter


class ONNXInference(InferenceBase):
    def __init__(self, model_loader, model_path, debug_mode=False):
        """
        Initialize the ONNXInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param model_path: Path to the ONNX model.
        :param debug_mode: If True, print additional debug information.
        """
        super().__init__(model_loader, onnx_path=model_path, debug_mode=debug_mode)

    def load_model(self):
        """
        Load the ONNX model. If the model does not exist, export it.

        :return: Loaded ONNX model.
        """
        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()
        return ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the ONNX model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark)

        input_name = self.model.get_inputs()[0].name
        ort_inputs = {input_name: input_data.cpu().numpy()}
        ort_outs = self.model.run(None, ort_inputs)

        # Extract probabilities from the output and normalize them
        if len(ort_outs) > 0:
            prob = ort_outs[0]
            if prob.ndim > 1:
                prob = prob[0]
            prob = np.exp(prob) / np.sum(np.exp(prob))
        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the ONNX model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
