from fastapi import FastAPI, Request
import uvicorn
import datetime
import os
import platform
from starlette.concurrency import run_in_threadpool


from src.utils import process_request

from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from detectaicore import index_response, Job

import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import sys
import traceback
from dotenv import load_dotenv


# load credentials
env_path = os.path.join("keys", ".env")
load_dotenv(env_path)
MODEL_PATH = os.getenv("MODEL_PATH")
LOCAL_ENV = os.getenv("LOCAL_ENV")


# Create APP
endpoint = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Create Jobs object
global jobs
jobs = {}

# output folder

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Setup HOME of app


output_folder = os.path.join(ROOT_DIR, "output")


if platform.system() == "Windows":
    MODELS_PATH = os.path.join(ROOT_DIR, "models")
elif platform.system() == "Linux" and MODEL_PATH == None and LOCAL_ENV == 0:
    MODEL_PATH = "app/models"
    MODELS_PATH = "app/models"
elif platform.system() == "Linux":
    MODELS_PATH = os.path.join(ROOT_DIR, "models")

print(MODELS_PATH)
try:
    model_path = os.path.join(MODELS_PATH, "nsfw_pytorch")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model = model.to(device)
except Exception as e:
    ex_type, ex_value, ex_traceback = sys.exc_info()
    print(f"Exception {ex_type} value {str(ex_value)}")


@endpoint.get("/test")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"test endpoint is": "OK"})


@endpoint.get("/health-check")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"message": "OK"})


@endpoint.get("/work/status")
async def status_handler(request: Request):
    return jobs


@endpoint.post("/process")
async def nsfw_process(request: Request, out: index_response):
    try:
        time1 = datetime.datetime.now()
        new_task = Job()
        # Capture Job and apply status
        jobs[new_task.uid] = new_task
        jobs[new_task.uid].status = "Job started"
        jobs[new_task.uid].type_job = "NSFW Model Analysis"
        response = await request.json()

        if isinstance(response.get("documents"), list):
            list_docs = response.get("documents")
        else:
            raise AttributeError("Expected a list of Documents")

        if isinstance(response.get("threshold"), str) or isinstance(
            response.get("threshold"), float
        ):
            if (
                isinstance(response.get("threshold"), str)
                and len(response.get("threshold")) > 0
            ):
                threshold = float(response.get("threshold"))
            elif isinstance(response.get("threshold"), float):
                threshold = response.get("threshold")
            else:
                threshold = 0.5
        # convert it into bytes
        data, documents_non_teathred = await run_in_threadpool(
            process_request,
            list_docs=list_docs,
            model=model,
            threshold=threshold,
            jobs=jobs,
            new_task=new_task,
            processor=processor,
            device=device,
        )

        # Print whole recognized text

        time2 = datetime.datetime.now()
        t = time2 - time1
        tf = t.seconds * 1000000 + t.microseconds

        # create response
        out.status = {"code": 200, "message": "Success"}
        out.data = data
        out.number_documents_treated = len(data)
        out.number_documents_non_treated = len(documents_non_teathred)
        out.list_id_not_treated = documents_non_teathred

        json_compatible_item_data = jsonable_encoder(out)
        # Update jobs interface
        jobs[new_task.uid].status = f"Job {new_task.uid} Finished code {200}"
        return JSONResponse(content=json_compatible_item_data, status_code=200)
    except Exception as e:
        # cath exception with sys and return the error stack
        out.status = {"code": 500, "message": "Error"}
        ex_type, ex_value, ex_traceback = sys.exc_info()
        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)

        # Format stacktrace
        stack_trace = list()

        for trace in trace_back:
            stack_trace.append(
                "File : %s , Line : %d, Func.Name : %s, Message : %s"
                % (trace[0], trace[1], trace[2], trace[3])
            )

        error = ex_type.__name__ + "\n" + str(ex_value) + "\n"
        for err in stack_trace:
            error = error + str(err) + "\n"
        out.error = error
        json_compatible_item_data = jsonable_encoder(out)
        return JSONResponse(content=json_compatible_item_data, status_code=500)


if __name__ == "__main__":
    uvicorn.run(
        "endpoint_nsfw:endpoint",
        host="127.0.0.1",
        port=5009,
        log_level="info",
    )
