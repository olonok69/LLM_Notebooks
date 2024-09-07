from torchvision import transforms
from PIL import Image
import torch


class ImageProcessor:
    def __init__(self, img_path: str, device: str = "cuda") -> None:
        """
        Initialize the ImageProcessor object.

        :param img_path: Path to the image to be processed.
        :param device: The device to process the image on ("cpu" or "cuda").
        """
        self.img_path = img_path
        self.device = device

    def process_image(self) -> torch.Tensor:
        """
        Process the image with the specified transformations: Resize, CenterCrop, ToTensor, and Normalize.

        :return: A batch of the transformed image tensor on the specified device.
        """
        # Open the image file
        img = Image.open(self.img_path)

        # Define the transformation pipeline
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # Apply transformations and prepare a batch
        img_transformed = transform(img)
        img_batch = torch.unsqueeze(img_transformed, 0).to(self.device)

        return img_batch
