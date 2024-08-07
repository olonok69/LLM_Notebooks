from detectaicore import Job, image_file_names, video_file_names
import numpy as np
import base64
import tempfile
from typing import List, Dict

# predict_video_frames
try:
    from image.nsfw.app.nsfw.inference import (
        predict_image_pytorch,
        predict_video_frames,
    )


except:
    from nsfw.inference import predict_image_pytorch, predict_video_frames


# this list it is purely image formats , diferent to what we use in OCR
image_file_names = ["jpeg", "jpg", "tiff", "png", "tif", "gif", "bmp"]


def analyse_video(data, model, processor, device, threshold):
    """
    Process video and get nfsw score
    params:
    data: list of documents from detect AI
    app global variables from FastAPI. contains tessetact and leptonica objects
    return:
    label classification and data dictionary
    """
    im_b64 = data.get("source").get("content")
    ext = data.get("source").get("file_type")
    file_name = data.get("source").get("file_name")
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(img_bytes)
        temp_filename = f.name

    print("Image saved to:", temp_filename)

    _, nsfw_probabilities = predict_video_frames(
        video_path=temp_filename, model=model, processor=processor, device=device
    )
    score = float(np.array(nsfw_probabilities).mean())
    if score > threshold:
        chunck = (
            f"video {file_name} is NOT Suitable for Work with a score of {str(score)}"
        )
        nsfw = 1

    else:
        chunck = f"video {file_name} is Suitable for Work with a score of {str(score)}"
        nsfw = 0
    print(chunck)

    data["source"]["content"] = [
        {"prediction": score, "description": chunck, "nsfw": nsfw}
    ]
    return chunck, data


def analyse_image(data, model, threshold, processor, device):
    """
    Process image and get nfsw score
    params:
    data: list of documents from detect AI
    app global variables from FastAPI. contains tessetact and leptonica objects
    return:
    label classification and data dictionary
    """
    im_b64 = data.get("source").get("content")
    ext = data.get("source").get("file_type")
    file_name = data.get("source").get("file_name")
    img_bytes = base64.b64decode(im_b64.encode("utf-8"))

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(img_bytes)
        temp_filename = f.name

    print("Image saved to:", file_name)

    try:
        # image_preds = classify(model, temp_filename, IMAGE_DIM)
        # preds = image_preds.get("data")
        image_preds = {}
        preds = predict_image_pytorch(temp_filename, processor, model, device)

        # if the aggregated probability of these 3 classes it is higher that threshold then NSFW = 1
        prob_nsfw = preds["sexy"] + preds["hentai"] + preds["porn"]
        if prob_nsfw > threshold:
            image_preds["nsfw"] = 1
        else:
            image_preds["nsfw"] = 0

        image_preds["class_probabilities"] = preds
        chunck = f"Image {file_name}, nsfw probability: {prob_nsfw}"
        print(chunck)
    except:
        image_preds = ""
        chunck = ""

    data["source"]["content"] = [image_preds]
    return chunck, data


def nsfw_analisys_documents_from_request(document, model, threshold, processor, device):
    """
    Process documents depending of their file type
    params:
    document: dictionary containing metadata and data of a document
    weights_path paths weights RestNet Model
    model keras model for Multiclass Classification
    return:
    label classification and data dictionary

    """
    # get extension of document from request
    ext = document.get("source").get("file_type")

    if ext in video_file_names:
        pass
        utf8_text, document = analyse_video(
            document, model, processor, device, threshold
        )
    elif ext in image_file_names:
        utf8_text, document = analyse_image(
            document, model, threshold, processor, device
        )

    return utf8_text, document


def process_request(
    list_docs: List[Dict],
    model,
    threshold: float,
    processor,
    device,
    jobs: Dict,
    new_task: Job,
):
    """ "
    process list of base64 Image/Documents NSFW model


    params:
    list_docs: list of documents from detect AI. List of dictionaries containing metadata and data of documents
    weights_path paths weights RestNet Model
    model keras model for Multiclass Classification
    jobs: dictionary to hold status of the Job
    new_task: Job object
    return:
    processed_docs : list of dictionaries containing document processed , this is pass trought NSFW model and text extracted. The extracted text replace the original base64 content
    documents_non_teathred : list of dictionaries containing {id : reason of not treating this id}
    """
    jobs[new_task.uid].status = "Start NSFW Analysis of Files"
    documents_non_teathred = []
    processed_docs = []
    for data in list_docs:
        file_type = data.get("source").get("file_type")
        file_name = data.get("source").get("file_name")
        print(
            f"Processing file : {file_name} length : {len(data.get('source').get('content'))}"
        )
        # if file type not valid
        if not file_type or not (file_type in image_file_names + video_file_names):
            documents_non_teathred.append({data.get("id"): "File type not valid"})
            print(f"File : {file_name}  No Valid Extension")
            continue

        utf8_text, data = nsfw_analisys_documents_from_request(
            data, model, threshold, processor, device
        )
        # if no text extracted doc to documents_non_teathred
        if len(utf8_text) < 3:
            documents_non_teathred.append(
                {data.get("id"): "No Image/Video suitable for NSFW analysis "}
            )
            print(f"File : {file_name}  No Text in this file")
            continue

        processed_docs.append(data)
        jobs[
            new_task.uid
        ].status = f"Extracted Text From {len(processed_docs)} Documents"
    return processed_docs, documents_non_teathred
