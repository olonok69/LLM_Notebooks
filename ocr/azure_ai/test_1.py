import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import time
import datetime
from IPython import embed
from dotenv import load_dotenv

env_path = os.path.join("keys", ".env")
load_dotenv(env_path)

# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.getenv("ENDPOINT")
    key = os.getenv("KEY")
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()


# Create an Image Analysis client
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


image_file = "data/senior_python_developer_nlplogix2_sm.jpg"

# Use Read API to read text in image
time1 = datetime.datetime.now()
with open(image_file, mode="rb") as image_data:
    # Wait for the asynchronous operation to complete

    read_results = client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ],
    )

    # Print text (OCR) analysis results to the console
    print(" Read:")
    if read_results.read is not None:
        for line in read_results.read.blocks[0].lines:
            print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
            # for word in line.words:
            #     print(
            #         f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}"
            #     )
time2 = datetime.datetime.now()
