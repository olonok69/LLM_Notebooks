import os
import numpy as np
import openvino as ov
from src.inference_base import InferenceBase
from src.onnx_exporter import ONNXExporter
from src.ov_exporter import OVExporter


class OVInference(InferenceBase):
    def __init__(self, model_loader, model_path, debug_mode=False):
        """
        Initialize the OVInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param model_path: Path to the OpenVINO model.
        :param debug_mode: If True, print additional debug information.
        """
        super().__init__(model_loader, ov_path=model_path, debug_mode=debug_mode)
        self.core = ov.Core()
        self.ov_model = self.load_model()
        self.compiled_model = self.core.compile_model(self.ov_model, "AUTO")

    def load_model(self):
        """
        Load the OpenVINO model. If the ONNX model does not exist, export it.

        :return: Loaded OpenVINO model.
        """
        # Determine the path for the ONNX model
        self.onnx_path = self.ov_path.replace(".ov", ".onnx")

        # Export ONNX model if it doesn't exist
        if not os.path.exists(self.onnx_path):
            onnx_exporter = ONNXExporter(
                self.model_loader.model, self.model_loader.device, self.onnx_path
            )
            onnx_exporter.export_model()

        ov_exporter = OVExporter(self.onnx_path)
        return ov_exporter.export_model()

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the OpenVINO model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark=is_benchmark)

        input_name = next(iter(self.compiled_model.inputs))
        outputs = self.compiled_model(inputs={input_name: input_data.cpu().numpy()})

        # Extract probabilities from the output and normalize them
        prob_key = next(iter(outputs))
        prob = outputs[prob_key]
        prob = np.exp(prob[0]) / np.sum(np.exp(prob[0]))

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the OpenVINO model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
