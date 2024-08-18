from fastapi import Request
from fastapi import FastAPI, Request
import uvicorn
import os
import json
import copy
import spacy
import sys
import traceback
from pathlib import Path
from starlette.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from detectaicore import (
    lfilenames_types,
    index_response,
    Job,
    image_file_names,
)
from src.utils import (
    extract_docs,
    get_pii_phi,
    get_pii_phi_v2,
    extract_pii_from_text,
    myconverter,
)
from src.logger_util import set_up_logging
from pii_codex.services.analysis_service import PIIAnalysisService
from pii_codex.config import (
    version,
    mapping_file_name,
    file_v1,
    MAX_LENGTH,
    APP_LANGUAGES,
)
from pii_codex.utils.pii_mapping_util import PIIMapper
from dotenv import load_dotenv
import logging


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
# load credentials
env_path = os.path.join("keys", ".env")
load_dotenv(env_path)

# Language Engine for this docker
if "LANGUAGE_ENGINE" in os.environ:
    LANGUAGE_ENGINE = os.getenv("LANGUAGE_ENGINE")
else:
    LANGUAGE_ENGINE = "en"
logging.info(f"Language this Analisys Engine {LANGUAGE_ENGINE}")

if "DOCKER" in os.environ:
    DOCKER = os.getenv("DOCKER")
else:
    DOCKER = "NO"
logging.info(f"I am working in a Docker Container {DOCKER}")
# load config dockers
if DOCKER == "NO":
    cfg_path = os.path.join(ROOT_DIR, "config", "config.json")
elif DOCKER == "YES":
    cfg_path = os.path.join(ROOT_DIR, "config", "config_docker.json")

config = json.load(open(cfg_path, "r"))
# Models path
MODEL_PATH = os.getenv("MODEL_PATH")
if MODEL_PATH == None:
    MODEL_PATH = "/home/detectai/models/classification"
logging.info(f"Environment model path {MODEL_PATH}")
# Setup Languages
sp = spacy.load("en_core_web_lg")
sp.max_length = MAX_LENGTH
# model specialized in NER
nlp = spacy.load("xx_ent_wiki_sm")
nlp.max_length = MAX_LENGTH
all_stopwords = sp.Defaults.stop_words
my_stop_words = [" "]


# PII Analyzer Engines
pii_analysys = {}
# initialize app engine
if LANGUAGE_ENGINE in APP_LANGUAGES:
    pii_analysys[LANGUAGE_ENGINE] = PIIAnalysisService(language_code=LANGUAGE_ENGINE)

logging.info(f"Configured Language Models {LANGUAGE_ENGINE} and NER")
# Create APP
endpoint = FastAPI()
endpoint.mapper_version = copy.deepcopy(version)
endpoint.file_name = copy.deepcopy(mapping_file_name)
endpoint.language_engine = LANGUAGE_ENGINE
endpoint.config = config
logging.info(f"Created FastAPI App")
# output folder

output_folder = os.path.join(ROOT_DIR, "output")

# Create Jobs object
global jobs
jobs = {}


@endpoint.get("/test")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"test endpoint is": "OK"})


@endpoint.get("/health-check")
async def health_check(request: Request):
    return JSONResponse(status_code=200, content={"message": "OK"})


@endpoint.get("/work/status")
async def status_handler(request: Request):
    return jobs


@endpoint.post("/process/process_text")
async def process_text(request: Request):
    """
    Process TIKA output received from Detect
    """

    response = await request.json()

    if isinstance(response.get("text"), str):
        text = response.get("text")
    else:
        raise AttributeError("Expected a Document")

    weights = response.get("weights")
    # only with version 2
    endpoint.mapper_version = "v2"
    mapper = None
    if isinstance(response.get("score"), str) or isinstance(
        response.get("score"), float
    ):
        if isinstance(response.get("score"), str) and len(response.get("score")) > 0:
            score = float(response.get("score"))
            score = score if score >= 0.4 else 0.4
        elif isinstance(response.get("score"), float):
            score = response.get("score")
            score = score if score >= 0.4 else 0.4
        else:
            score = 0.4
    # personalized mapper or default mapper
    if len(weights) > 0 and endpoint.mapper_version == "v2":
        logging.info(f"V2 Use Custom Weights. Score = {score}")
        del mapper
        mapper = PIIMapper(
            version=endpoint.mapper_version,
            mapping_file_name=endpoint.file_name,
            test=False,
            reload=True,
            weigths=weights,
        )
    elif len(weights) == 0 and endpoint.mapper_version == "v2":
        logging.info(f"V2 Use Default Weights. Score = {score}")
        del mapper
        mapper = PIIMapper(
            version=endpoint.mapper_version,
            mapping_file_name=endpoint.file_name,
            test=False,
            reload=False,
        )

    lang = endpoint.language_engine
    pii_analysys_engine = pii_analysys.get(lang)

    pii_analysys_engine.pii_mapper = mapper
    pii_analysys_engine._pii_assessment_service.pii_mapper = mapper
    pii_analysys_engine._analyzer.pii_mapper = mapper
    # logging
    logging.info(f"Processing Text Engine {LANGUAGE_ENGINE} Length {len(text)}")
    logging.info(f"Processing Text Engine {LANGUAGE_ENGINE} Score: {score}")
    logging.info(
        f"Processing Text Engine {LANGUAGE_ENGINE} Version: {endpoint.mapper_version}"
    )
    logging.info(
        f"Processing Text Engine {LANGUAGE_ENGINE} Length Weights: {len(weights)}"
    )
    analysis_results = await extract_pii_from_text(
        pii_analysys_engine, lang=lang, score=score, text=text
    )
    # convertir json to compatible
    json_compatible_item_data = json.dumps(analysis_results, default=myconverter)
    json_compatible_item_data = json.loads(json_compatible_item_data)
    return JSONResponse(content=json_compatible_item_data, status_code=200)


@endpoint.post("/process")
async def process_tika(request: Request, out: index_response):
    """
    Process TIKA output received from Detect
    """

    try:
        response = await request.json()
        mapper = None
        new_task = Job()
        # Capture Job and apply status
        jobs[new_task.uid] = new_task
        jobs[new_task.uid].status = "Job started"
        jobs[new_task.uid].type_job = "Classification Model Analysis"

        if isinstance(response.get("documents"), list):
            list_docs = response.get("documents")
        else:
            raise AttributeError("Expected a list of Documents")

        weights = response.get("weights")
        ocr = response.get("ocr")
        # Extract response elements
        if isinstance(response.get("ocr"), int) or isinstance(response.get("ocr"), str):
            if isinstance(response.get("ocr"), str):
                ocr = int(response.get("ocr"))
            elif isinstance(response.get("ocr"), int):
                ocr = response.get("ocr")
            else:
                ocr = 0
        else:
            ocr = 0
        if isinstance(response.get("weights"), str) or isinstance(
            response.get("weights"), list
        ):
            if isinstance(response.get("weights"), list):
                weights = response.get("weights")
            else:
                weights = ""
        else:
            weights = ""

        if isinstance(response.get("score"), str) or isinstance(
            response.get("score"), float
        ):
            if (
                isinstance(response.get("score"), str)
                and len(response.get("score")) > 0
            ):
                score = float(response.get("score"))
                score = score if score >= 0.4 else 0.4
            elif isinstance(response.get("score"), float):
                score = response.get("score")
                score = score if score >= 0.4 else 0.4
            else:
                score = 0.4

        if isinstance(response.get("version"), str):
            if (
                isinstance(response.get("version"), str)
                and len(response.get("version")) > 0
            ):
                version = response.get("version")
                if version == "v1" or version == "v2":
                    endpoint.mapper_version = version
                elif int(os.environ.get("IS_TEST", "0")) == 1:
                    endpoint.mapper_version = "v1"
                else:
                    endpoint.mapper_version = "v2"

            else:
                endpoint.mapper_version = "v2"
        elif int(os.environ.get("IS_TEST", "0")) == 1:
            endpoint.mapper_version = "v1"
        else:
            endpoint.mapper_version = "v2"

        # personalized mapper or default mapper
        if len(weights) > 0 and endpoint.mapper_version == "v2":
            logging.info(f"V2 Use Custom Weights. Score = {score}")
            del mapper
            mapper = PIIMapper(
                version=endpoint.mapper_version,
                mapping_file_name=endpoint.file_name,
                test=False,
                reload=True,
                weigths=weights,
            )
        elif len(weights) == 0 and endpoint.mapper_version == "v2":
            logging.info(f"V2 Use Default Weights. Score = {score}")
            del mapper
            mapper = PIIMapper(
                version=endpoint.mapper_version,
                mapping_file_name=endpoint.file_name,
                test=False,
                reload=False,
            )
        elif endpoint.mapper_version == "v1":
            print(f"V1. Score = {score}")
            del mapper
            mapper = PIIMapper(
                version=endpoint.mapper_version,
                mapping_file_name=file_v1,
                test=False,
                reload=False,
            )
        logging.info(
            f"Processing Front API PII Model Engine. Number of Documents {len(list_docs)}"
        )
        logging.info(f"Processing Front API PII Model Engine. Score: {score}")
        logging.info(
            f"Processing Front API PII Model Engine. Version: {endpoint.mapper_version}"
        )
        logging.info(
            f"Processing Front API PII Model Engine. Length Weights: {len(weights)}"
        )
        logging.info(f"Start extract Docs method. Extracting data from Tika Documents")

        # Extract Metadata
        docs_with_languages, documents_non_processed = await run_in_threadpool(
            extract_docs,
            list_docs=list_docs,
            list_pii_docs=[],
            jobs=jobs,
            new_task=new_task,
            file_types_all=False,
            filenames_types=lfilenames_types,
            image_file_names=image_file_names,
            ocr=ocr,
            version=endpoint.mapper_version,
        )
        # Extract individual PII/PHI from documents
        logging.info(
            f"Number of Documents extracted to process in Analisys Engine = {len(docs_with_languages)}"
        )

        if len(docs_with_languages) == 0:
            out.status = {
                "code": 500,
                "message": "Error",
            }
            out.data = []
            out.error = "No valid documents in these batch"
            json_compatible_item_data = jsonable_encoder(out)
            return JSONResponse(content=json_compatible_item_data, status_code=500)

        if endpoint.mapper_version == "v1":
            logging.info(f"Start Analisys Process with Risk Model Version v1")
            chunck, _, documents_non_processed = await get_pii_phi(
                pii_analysys,
                nlp,
                docs_with_languages,
                documents_non_processed,
                all_stopwords,
                my_stop_words,
                jobs=jobs,
                new_task=new_task,
                pii_mapper=mapper,
                score=score,
            )
        elif endpoint.mapper_version == "v2":
            logging.info(f"Start Analisys Process with Risk Model Version v2")
            chunck, _, documents_non_processed = await get_pii_phi_v2(
                nlp=nlp,
                docs_with_languages=docs_with_languages,
                documents_non_teathred=documents_non_processed,
                all_stopwords=all_stopwords,
                my_stop_words=my_stop_words,
                config=endpoint.config,
                jobs=jobs,
                new_task=new_task,
                score=score,
                weights=weights,
            )
        logging.info(f"Finish Analisys Process")
        logging.info(f"Number of documents Received {len(list_docs)}")
        logging.info(
            f"Number of documents sucessfully analyzed {len(docs_with_languages)}"
        )
        logging.warn(f"Number of documents NO analyzed {len(documents_non_processed)}")
        # logging reasons
        for d in documents_non_processed:
            for key, value in d.items():
                logging.warn(f"Document {key} non processed, Reason {value}")

        out.status = {"code": 200, "message": "Success"}
        out.data = chunck
        out.number_documents_treated = len(docs_with_languages)
        out.number_documents_non_treated = len(documents_non_processed)
        out.list_id_not_treated = documents_non_processed
        if len(documents_non_processed) == 0:
            out.error = "Batch Processed without error"
        else:
            out.error = (
                f"Batch Processed with {len(documents_non_processed)} non processed"
            )
        json_compatible_item_data = jsonable_encoder(out)
        # update Job Status
        jobs[new_task.uid].status = f"Job {new_task.uid} Finished"
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
    # reload=True if need it
    if ("DOCKER" not in os.environ) or (
        "DOCKER" in os.environ and os.getenv("DOCKER") == "NO"
    ):

        uvicorn.run(
            "endpoint_classification:endpoint",
            host="127.0.0.1",
            port=5000,
            log_level="info",
        )
