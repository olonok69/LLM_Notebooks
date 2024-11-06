from fastapi import Request
from fastapi import FastAPI, Request
import uvicorn
import os
import json
from pathlib import Path
from fastapi.encoders import jsonable_encoder
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from detectaicore import index_response, set_up_logging, print_stack
from typing import List, Optional
import gc
import torch
from dotenv import load_dotenv
import platform
import warnings
import logging
import copy
from src.training import (
    create_tokenizer,
    tokenize_datasets,
    create_model_sequence_classification,
    create_trainer,
    train_model,
    predict,
    clean_gpu,
)
from src.utils_mlflow import (
    get_all_experiments_last_run,
    get_mlflow_job_status,
    delete_experiment_soft,
    delete_run_id_soft,
    delete_experiment_hard,
)
from src.utils_data import (
    create_dataframe_full,
    validate_validation_dataset,
)

warnings.filterwarnings("ignore")
# logging
# Set up logging
LOGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
Path(LOGS_PATH).mkdir(parents=True, exist_ok=True)
script_name = os.path.join(LOGS_PATH, "debug.log")
# create loggers
if not set_up_logging(
    console_log_output="stdout",
    console_log_level="info",
    console_log_color=True,
    logfile_file=script_name,
    logfile_log_level="debug",
    logfile_log_color=False,
    log_line_template="%(color_on)s[%(asctime)s] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s",
):
    print("Failed to set up logging, aborting.")
    raise AttributeError("failed to create logging")

# Load rest of parameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# load credentials
env_path = os.path.join("config", ".env")
load_dotenv(env_path)
# env variables
MODEL_PATH = os.getenv("MODEL_PATH")
LOCAL_ENV = os.getenv("LOCAL_ENV")
USER = os.getenv("USER")
# load password to access mysql from os.environment
if "MYSQL_PASSWORD" not in os.environ:
    logging.warning(
        "MYSQL_PASSWORD not in environtment. It will be issues deleting Experiments"
    )
PASSWORD = os.getenv("MYSQL_PASSWORD")

# Load rest of parameters
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# load config
config_path = os.path.join(ROOT_DIR, "config", "config.json")
config = json.load(open(config_path))
MLFLOW_URI = config.get("MLFLOW_URI")
EXPERIMENT_NAME = config.get("EXPERIMENT_NAME")
MODEL_ID = config.get("MODEL_ID")
CLASSIFIER_TYPE = config.get("CLASSIFIER_TYPE")
# LIMITS DATASET
SIZE_TRAIN = config.get("SIZE_TRAIN")
MIN_SIZE_LABEL = config.get("MIN_SIZE_LABEL")
# DATABASE
HOST = config.get("HOST")
USER_DB = config.get("USER_DB")
PORT = config.get("PORT")
DB = config.get("DB")
# DEVICE
DEVICE = config.get("DEVICE")
# Log configuration values
for k, v in config.items():
    logging.info(f"Configuration key: {k}, value: {v}")
# run on local
if platform.system() == "Windows" or (
    platform.system() == "Linux" and USER == "olonok"
):
    MLFLOW_URI = "http://127.0.0.1:5000"
    HOST = "127.0.0.1"
logging.info(f"Host: {HOST} and MLFLOW_URI: {MLFLOW_URI}")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(ROOT_DIR, "output")

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

# Create APP
endpoint = FastAPI()
# Clean GPU
gc.collect()
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() and DEVICE == "cuda" else "cpu"
logging.info(f"Device for training: {device}")
# Create Jobs object
global jobs, status
jobs, status = {}, {}


@endpoint.get("/test")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"test endpoint is": "OK"})


@endpoint.get("/health-check")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"message": "OK"})


@endpoint.get("/work/status")
async def status_handler(request: Request):
    return jobs


@endpoint.post("/job_status")
async def post_job_status(request: Request):
    """ "
    Get status via post
    data = {}
    data["experiment_name"] = "Custom_Classifier3"
    data1 = json.dumps(data).encode()

    url = "http://localhost:5021/jobs_status"
    r = requests.post(url, data=data1)

    """
    response = await request.json()
    if isinstance(response.get("experiment_name"), str):
        experiment_name = response.get("experiment_name")
        logging.info(f"Get runs of Experiment {experiment_name}")
    else:
        raise AttributeError(
            "Expected a experiment Name where to Log outputs of experiment the data"
        )

    out = get_mlflow_job_status(mlflow_uri=MLFLOW_URI, experiment_name=experiment_name)
    return JSONResponse(status_code=200, content=json.loads(out))


@endpoint.get("/runid/status")
async def status_handler_jobs(request: Request, experiment_name: Optional[str] = None):
    """
    Get status via get
    url = "http://localhost:5021/runid/status"
    r = requests.get(url, params={"experiment_name": "Custom_Classifier3"})

    """
    if isinstance(experiment_name, str):
        logging.info(f"Get runs of Experiment {experiment_name}")
    else:
        raise AttributeError(
            "Expected a experiment Name where to Log outputs of experiment the data"
        )
    out = get_mlflow_job_status(mlflow_uri=MLFLOW_URI, experiment_name=experiment_name)
    return json.loads(out)


@endpoint.get("/all_experiments/last_run")
async def all_experiments_last_run(experiment_name: Optional[str] = None):
    """
    Get status via get
    url = "http://localhost:5020/all_experiments/last_run"
    r = requests.get(url)

    """
    # set tracking URI
    status = get_all_experiments_last_run(MLFLOW_URI)
    return json.loads(status)


@endpoint.post("/experiment/soft_delete")
async def post_soft_delete_experiment(request: Request):
    """ "
    delete soft Experiment
    data = {}
    data["experiment_name"] = "Custom_Classifier3"
    data1 = json.dumps(data).encode()

    url = "http://localhost:5021/experiment/soft_delete"
    r = requests.post(url, data=data1)

    """
    response = await request.json()
    if isinstance(response.get("experiment_name"), str):
        experiment_name = response.get("experiment_name")
        logging.info(f"Soft delete Experiment {experiment_name}")
    else:
        raise AttributeError(
            "Expected a experiment Name where to Log outputs of experiment the data"
        )

    out = delete_experiment_soft(mlflow_uri=MLFLOW_URI, experiment_name=experiment_name)

    return JSONResponse(status_code=200, content=out)


@endpoint.post("/run/soft_delete")
async def post_soft_delete_run_id(request: Request):
    """ "
    delete soft runID
    data = {}
    data["run_id"] = "1fca3d03b0e847eb859ebc49235639dd"
    data1 = json.dumps(data).encode()

    url = "http://localhost:5021/run/soft_delete"
    r = requests.post(url, data=data1)

    """
    response = await request.json()
    if isinstance(response.get("run_id"), str):
        run_id = response.get("run_id")
        logging.info(f"Soft delete runID {run_id}")
    else:
        raise AttributeError(
            "Expected a run_id Name where to Log outputs of experiment the data"
        )

    out = delete_run_id_soft(mlflow_uri=MLFLOW_URI, run_id=run_id)

    return JSONResponse(status_code=200, content=out)


@endpoint.post("/experiment/hard_delete")
async def post_hard_delete_experiment(request: Request, out: index_response):
    """ "
    delete hard Experiment
    data = {}
    data["experiment_name"] = "Custom_Classifier3"
    data1 = json.dumps(data).encode()

    url = "http://localhost:5021/experiment/hard_delete"
    r = requests.post(url, data=data1)

    """
    response = await request.json()
    try:
        if isinstance(response.get("experiment_name"), str):
            experiment_name = response.get("experiment_name")
            logging.info(f"Hard delete Experiment {experiment_name}")
        else:
            raise AttributeError(
                "Expected a experiment Name where to Log outputs of experiment the data"
            )
        # delete experiment deleted
        out = delete_experiment_hard(
            mlflow_uri=MLFLOW_URI,
            experiment_name=experiment_name,
            host=HOST,
            user=USER_DB,
            port=PORT,
            password=PASSWORD,
            db=DB,
        )

        return JSONResponse(status_code=200, content=out)
    except Exception:
        # cath exception with sys and return the error stack
        json_compatible_item_data = print_stack(out)
        return JSONResponse(content=json_compatible_item_data, status_code=500)


@endpoint.post("/process")
async def process_customer_classifier(request: Request, out: index_response):
    """
    Process start custom classifier
    """

    try:
        response = await request.json()

        if isinstance(response.get("labels"), List):
            labels = response.get("labels")
        else:
            raise AttributeError("Expected a list of labels")

        if isinstance(response.get("path"), str):
            path = response.get("path")
        else:
            logging.error("Expected a path where to find the data")
            raise AttributeError("Expected a path where to find the data")

        if isinstance(response.get("experiment_name"), str):
            EXPERIMENT_NAME = response.get("experiment_name")
        else:
            logging.error(
                "Expected a experiment Name where to Log outputs of experiment the data"
            )
            raise AttributeError(
                "Expected a experiment Name where to Log outputs of experiment the data"
            )
        if isinstance(response.get("validation"), int) or isinstance(
            response.get("validation"), str
        ):
            VALIDATION = int(response.get("validation"))
        else:
            VALIDATION = 0
        for label in labels:
            logging.info(f"Label in this set: {label}")
        logging.info(f"Path Dataset: {path}")
        logging.info(f"Experiment Name: {EXPERIMENT_NAME}")
        logging.info(f"Validation Dataset 1=True 0=False: {VALIDATION}")
        # create tokenizer
        endpoint.tokenizer = create_tokenizer(MODEL_ID)

        DATAPATH = os.path.join(ROOT_DIR, path)  # Change to ROOT_DIR
        # phase1 prepare dataset
        logging.info("Phase1 prepare dataset")
        (
            run,
            train_dataset_transformers,
            test_dataset_transformers,
            validation_dataset_transformers,
            label2id,
            id2label,
            list_sizes,
            list_sizes_rest,
            min_size_rest,
        ) = await run_in_threadpool(
            create_dataframe_full,
            DATAPATH,
            labels,
            min_num_samples=MIN_SIZE_LABEL,
            size_train=SIZE_TRAIN,
            mlflow_uri=MLFLOW_URI,
            experiment_name=EXPERIMENT_NAME,
            output_folder=OUTPUT,
            validation=VALIDATION,
        )
        # Phase 2 tokenize Datasets
        logging.info("Phase 2 tokenize Datasets")
        train_tokenized, test_tokenized = await run_in_threadpool(
            tokenize_datasets,
            train_dataset_transformers,
            test_dataset_transformers,
            endpoint.tokenizer,
        )
        # If validation is on
        if VALIDATION == 1:
            logging.info("Validation activated")
            list_sizes = validate_validation_dataset(
                data=validation_dataset_transformers,
                labels=labels,
                label2id=label2id,
                min_size_label=MIN_SIZE_LABEL,
            )

            logging.info(f"Validation Dataset class sizes {list_sizes}")
        # Phase 3 create model
        logging.info("Phase 3 create model")
        endpoint.model = None
        if CLASSIFIER_TYPE == "text-classifier":
            endpoint.model = await run_in_threadpool(
                create_model_sequence_classification,
                MODEL_ID,
                label2id,
                id2label,
                device=device,
            )
            logging.info("Phase 3 create trainer")
            # create Trainer object
            endpoint.trainer = await run_in_threadpool(
                create_trainer,
                endpoint,
                train_tokenized,
                test_tokenized,
                labels,
                device=device,
            )
            dataset_val = None
            if int(VALIDATION) == 1:
                dataset_val = copy.deepcopy(validation_dataset_transformers)
            run, model_info = await run_in_threadpool(
                train_model,
                endpoint=endpoint,
                mlflow_uri=MLFLOW_URI,
                experiment_name=EXPERIMENT_NAME,
                labels=labels,
                label2id=label2id,
                validation=VALIDATION,
                validation_dataset=dataset_val,
            )
            logging.info(model_info.model_uri)

        dict_run = run.to_dictionary()

        # clean memory
        if device == "cuda":
            clean_gpu(endpoint)

        docs_with_languages = []
        documents_non_processed = []
        out.status = {"code": 200, "message": "Success"}
        out.data = dict_run
        out.number_documents_treated = len(docs_with_languages)
        out.number_documents_non_treated = len(documents_non_processed)
        out.list_id_not_treated = documents_non_processed

        json_compatible_item_data = jsonable_encoder(out)
        return JSONResponse(content=json_compatible_item_data, status_code=200)

    except Exception:
        # cath exception with sys and return the error stack
        json_compatible_item_data = print_stack(out)
        return JSONResponse(content=json_compatible_item_data, status_code=500)


@endpoint.post("/predict")
async def predict_customer_classifier(request: Request, out: index_response):

    try:
        response = await request.json()

        if isinstance(response.get("runid"), str):
            runid = response.get("runid")
        else:
            logging.error("Expected a valid mlflow runid")
            raise AttributeError("Expected a valid mlflow runid")

        if isinstance(response.get("text"), str) and len(response.get("text")) > 10:
            text = response.get("text")
        else:
            logging.error("EExpected text to classify")
            raise AttributeError("Expected text to classify")

        dict_run = predict(runid=runid, text=text, mlflow_uri=MLFLOW_URI, device=device)

        docs_with_languages = []
        documents_non_processed = []
        out.status = {"code": 200, "message": "Success"}
        out.data = dict_run
        out.number_documents_treated = len(docs_with_languages)
        out.number_documents_non_treated = len(documents_non_processed)
        out.list_id_not_treated = documents_non_processed

        json_compatible_item_data = jsonable_encoder(out)
        return JSONResponse(content=json_compatible_item_data, status_code=200)
    except Exception:
        # cath exception with sys and return the error stack
        json_compatible_item_data = print_stack(out)
        return JSONResponse(content=json_compatible_item_data, status_code=500)


if __name__ == "__main__":
    USER = os.getenv("USER")
    if platform.system() == "Windows" or (
        platform.system() == "Linux" and USER == "olonok"
    ):

        uvicorn.run(
            "endpoint_custom_classifier:endpoint",
            host="127.0.0.1",
            reload=True,
            port=5020,
            log_level="info",
        )
