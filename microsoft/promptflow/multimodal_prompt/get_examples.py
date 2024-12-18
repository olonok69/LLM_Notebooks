from promptflow.core import tool
import base64, requests
import os

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
    return "data:image/tif;base64," + base64_string


@tool
def get_examples():
    """
    Returns a list of example images and their categories."""
    check = encode_image_to_base64(os.path.join(ROOT_DIR, "images/check.tif"))
    fax = encode_image_to_base64(os.path.join(ROOT_DIR, "images/fax.tif"))
    invoice = encode_image_to_base64(os.path.join(ROOT_DIR, "images/invoice.tif"))

    return [
        {"category": "check", "image": check},
        {"category": "fax", "image": fax},
        {"category": "invoice", "image": invoice},
    ]
