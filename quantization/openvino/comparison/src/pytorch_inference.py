import torch
from src.inference_base import InferenceBase


class PyTorchInference(InferenceBase):
    def __init__(self, model_loader, device="cpu", debug_mode=False):
        """
        Initialize the PyTorchInference object.

        :param model_loader: Object responsible for loading the model and categories.
        :param device: The device to load the model on ("cpu" or "cuda").
        :param debug_mode: If True, print additional debug information.
        """
        self.device = device
        super().__init__(model_loader, debug_mode=debug_mode)
        self.model = self.load_model()

    def load_model(self):
        """
        Load the PyTorch model to the specified device.

        :return: Loaded PyTorch model.
        """
        return self.model_loader.model.to(self.device)

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data using the PyTorch model.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        :return: Top predictions based on the probabilities.
        """
        super().predict(input_data, is_benchmark=is_benchmark)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_data.to(self.device))

        # Compute the softmax probabilities
        prob = torch.nn.functional.softmax(outputs[0], dim=0)
        prob = prob.cpu().numpy()

        return self.get_top_predictions(prob, is_benchmark)

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance using the PyTorch model.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        return super().benchmark(input_data, num_runs, warmup_runs)
