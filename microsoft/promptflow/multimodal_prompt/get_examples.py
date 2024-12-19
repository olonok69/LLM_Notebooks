from promptflow.core import tool
import base64, requests
import os
from rapidocr_onnxruntime import RapidOCR

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def url_to_base64(image_url):
    response = requests.get(image_url)
    return "data:image/jpg;base64," + base64.b64encode(response.content).decode("utf-8")


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file in binary mode
        image_data = image_file.read()
        # Encode the binary data to Base64
        base64_encoded_data = base64.b64encode(image_data)
        # Convert the Base64 bytes to a string
        base64_string = base64_encoded_data.decode("utf-8")
    return "data:image/png;base64," + base64_string


def extract_text(image_path, engine):
    result, _ = engine(image_path)
    content = ""
    for r in result:
        content = content + " " + r[1]
    return content


@tool
def get_examples():
    """
    Returns a list of example images and their categories."""
    engine = RapidOCR()

    check = extract_text(
        image_path=os.path.join(ROOT_DIR, "images/check.png"), engine=engine
    )
    fax = extract_text(
        image_path=os.path.join(ROOT_DIR, "images/fax.png"), engine=engine
    )
    invoice = extract_text(
        image_path=os.path.join(ROOT_DIR, "images/invoice.png"), engine=engine
    )

    return [
        {"category": "check", "image": check},
        {"category": "fax", "image": fax},
        {"category": "invoice", "image": invoice},
    ]
