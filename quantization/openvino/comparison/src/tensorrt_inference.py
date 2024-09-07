import torch
# import torch_tensorrt
import logging
from src.inference_base import InferenceBase
import torch_tensorrt

# Check for CUDA and TensorRT availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    try:
        import torch_tensorrt as trt
    except ImportError:
        logging.warning("torch-tensorrt is not installed. Running on CPU mode only.")
        CUDA_AVAILABLE = False


class TensorRTInference(InferenceBase):
    def __init__(self, model_loader, device, precision=torch.float32, debug_mode=False):
        """
        Initialize the TensorRTInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param precision: Precision mode for TensorRT (default is torch.float32).
        """
        self.precision = precision
        self.device = device
        super().__init__(model_loader, debug_mode=debug_mode)
        if CUDA_AVAILABLE:
            self.load_model()

    def load_model(self):
        """
        Load and convert the PyTorch model to TensorRT format.
        """
        # Load the PyTorch model
        self.model = self.model_loader.model.to(self.device).eval()

        # Convert the PyTorch model to TorchScript
        scripted_model = torch.jit.trace(
            self.model, torch.randn((1, 3, 224, 224)).to(self.device)
        )

        # Compile the TorchScript model with TensorRT
        if CUDA_AVAILABLE:
            self.model = torch_tensorrt.compile(
                scripted_model,
                ir="torchscript",
                inputs=[torch_tensorrt.Input((1, 3, 224, 224), dtype=self.precision)],
                enabled_precisions={self.precision},
            )

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the TensorRT model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark=is_benchmark)

        with torch.no_grad():
            outputs = self.model(input_data.to(self.device).to(dtype=self.precision))

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the TensorRT model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
