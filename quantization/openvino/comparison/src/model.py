import pandas as pd
from torchvision import models


class ModelLoader:
    def __init__(self, model_type: str = "resnet50", device: str = "cuda") -> None:
        """
        Initialize the ModelLoader object.

        :param model_type: Type of the model to load ("resnet50", "efficientnet", etc.).
        :param device: The device to load the model on ("cpu" or "cuda").
        """
        self.device = device
        self.model = self.load_model(model_type)
        self.categories: pd.DataFrame = pd.read_csv(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
            header=None,
        )

    def load_model(self, model_type: str):
        if model_type == "resnet50":
            return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(self.device)
        elif model_type == "efficientnet":
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1).to(self.device)
        elif model_type == "efficientnet_b7":
            return models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1).to(self.device)
        elif model_type == "mobilenet_v2":
            return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
