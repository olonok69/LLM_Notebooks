import os
import openvino as ov


class OVExporter:
    """
    OVExporter handles the conversion of an ONNX model to OpenVINO's internal representation.
    """

    def __init__(self, onnx_model_path: str):
        """
        Initialize the OVExporter with the path to the ONNX model.

        :param onnx_model_path: str
            Path to the ONNX model file.
        """
        self.onnx_path = onnx_model_path
        self.core = ov.Core()

    def export_model(self) -> ov.Model:
        """
        Convert the ONNX model to OpenVINO's internal representation.

        :return: ov.ie.IENetwork
            The converted OpenVINO model.
        """
        if not os.path.isfile(self.onnx_path):
            raise ValueError(f"ONNX model wasn't found in path: {self.onnx_path}")

        # Convert the ONNX model to OpenVINO's internal representation
        ov_model = self.core.read_model(self.onnx_path)
        return ov_model
