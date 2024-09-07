import time
import logging
import numpy as np
import torch


class InferenceBase:
    def __init__(
        self,
        model_loader,
        onnx_path=None,
        ov_path=None,
        topk=5,
        debug_mode=False,
        batch_size=8,
    ):
        """
        Base class for inference.

        :param model_loader: Object responsible for loading the model and categories.
        :param onnx_path: Path to the ONNX model (if applicable).
        :param ov_path: Path to the OpenVINO model (if applicable).
        :param topk: Number of top predictions to return.
        :param debug_mode: If True, print additional debug information.
        :param batch_size: How many input images to stack for benchmark
        """
        self.model_loader = model_loader
        self.onnx_path = onnx_path
        self.ov_path = ov_path
        self.categories = model_loader.categories
        self.model = self.load_model()
        self.topk = topk
        self.debug_mode = debug_mode
        self.batch_size = batch_size

    def load_model(self):
        """
        Load the model. This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def predict(self, input_data, is_benchmark=False):
        """
        Run prediction on the input data.

        :param input_data: Data to run the prediction on.
        :param is_benchmark: If True, the prediction is part of a benchmark run.
        """
        if not is_benchmark:
            logging.info(f"Running prediction for {self.__class__.__name__} model")
            if self.debug_mode:
                print(f"Running prediction for {self.__class__.__name__} model")

    def benchmark(self, input_data, num_runs=100, warmup_runs=50):
        """
        Benchmark the prediction performance.

        :param input_data: Data to run the benchmark on.
        :param num_runs: Number of runs for the benchmark.
        :param warmup_runs: Number of warmup runs before the benchmark.
        :return: Average inference time in milliseconds.
        """
        # Expand batch size to stack identical images to load the system for benchmark
        if len(input_data.shape) == 4:
            input_data = input_data.squeeze(0)
        input_batch = torch.stack([input_data] * self.batch_size)

        # Warmup
        logging.info(f"Starting warmup for {self.__class__.__name__} inference...")
        for _ in range(warmup_runs):
            for img in input_batch:
                self.predict(img.unsqueeze(0), is_benchmark=True)

        # Benchmark
        logging.info(f"Starting benchmark for {self.__class__.__name__} inference...")
        start_time = time.time()
        for _ in range(num_runs):
            for img in input_batch:
                self.predict(img.unsqueeze(0), is_benchmark=True)
        avg_time = (
            (time.time() - start_time) / (num_runs * self.batch_size)
        ) * 1000  # Convert to ms

        logging.info(f"Average inference time for {num_runs} runs: {avg_time:.4f} ms")
        if self.debug_mode:
            print(
                f"Average inference time for {self.__class__.__name__} and {num_runs} runs: {avg_time:.4f} ms"
            )

        # Calculate throughput
        total_samples = input_data.size(0) * num_runs
        total_time_seconds = time.time() - start_time
        throughput = total_samples / total_time_seconds

        logging.info(
            f"Throughput for {self.__class__.__name__}: {throughput:.2f} samples/sec"
        )
        if self.debug_mode:
            print(
                f"Throughput for {self.__class__.__name__}: {throughput:.2f} samples/sec"
            )

        return avg_time, throughput

    def get_top_predictions(self, prob: np.ndarray, is_benchmark=False):
        """
        Get the top predictions based on the probabilities.

        :param prob: Array of probabilities.
        :param is_benchmark: If True, the method is called during a benchmark run.
        :return: Array of probabilities.
        """
        if is_benchmark:
            return None

        # Get the top indices and probabilities
        top_indices = prob.argsort()[-self.topk :][::-1]
        top_probs = prob[top_indices]

        # Log and print the top predictions
        for i in range(self.topk):
            probability = top_probs[i]
            class_label = self.categories[0][int(top_indices[i])]
            logging.info(f"#{i + 1}: {int(probability * 100)}% {class_label}")
            if self.debug_mode:
                print(f"#{i + 1}: {int(probability * 100)}% {class_label}")
        return prob
