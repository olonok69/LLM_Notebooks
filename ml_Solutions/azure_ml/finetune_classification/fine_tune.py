## Request Quotas
# https://learn.microsoft.com/en-gb/azure/machine-learning/how-to-manage-quotas?view=azureml-api-2#request-quota-increases

## Log Metrics
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2&tabs=interactive#view-jobsruns-information-in-the-studio

## Dataset
# https://huggingface.co/datasets/dair-ai/emotion


# https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.mlclient?view=azure-python

###%pip install azure-ai-ml azure-identity datasets==2.9.0 mlflow azureml-mlflow

### Example Llama2
# https://balabala76.medium.com/llama-2-fine-tuning-using-azure-machine-learning-in-parallel-cf9720d1d60e

# import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from dotenv import dotenv_values
from dotenv import load_dotenv
from src.login import get_ws_client
from src.datasets import get_labels_dataset, create_datasets
from src.computer import create_gpu_cluster
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import os
import mlflow
import time
import pandas as pd
import json
import ast
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    ProbeSettings,
)
from IPython import embed

# where I am
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "emotion-dataset")

# Load env and login to Workspace
load_dotenv(".env")
config = dotenv_values(".env")

# Enter details of your Azure Machine Learning workspace
subscription_id = config.get("SUBSCRIPTION_ID")
resource_group = config.get("RESOURCE_GROUP")
workspace = config.get("AZUREML_WORKSPACE_NAME")


credential = DefaultAzureCredential()
# Check if given credential can get token successfully.
credential.get_token("https://management.azure.com/.default")


workspace_ml_client = get_ws_client(
    credential, subscription_id, resource_group, workspace
)
print(workspace_ml_client)

# the models, fine tuning pipelines and environments are available in the AzureML system registry, "azureml"
registry_ml_client = MLClient(credential, registry_name="azureml")
print(registry_ml_client)
registry_ml_client_meta = MLClient(credential, registry_name="azureml-meta")
print(registry_ml_client_meta)


experiment_name = "text-classification-emotion-detection"

# generating a unique timestamp that can be used for names and versions that need to be unique
timestamp = str(int(time.time()))

# Pick a foundation model to fine tune

model_name = "Llama-2-7b"  # registry_ml_client
foundation_model = registry_ml_client_meta.models.get(model_name, label="latest")
# If not Llama use registry_ml_client
model_name = "bert-base-uncased"  # registry_ml_client
foundation_model = registry_ml_client.models.get(model_name, label="latest")

print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for fine tuning".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)


if "computes_allow_list" in foundation_model.tags:
    computes_allow_list = ast.literal_eval(
        foundation_model.tags["computes_allow_list"]
    )  # convert string to python list
    print(f"Please create a compute from the above list - {computes_allow_list}")
else:
    computes_allow_list = None
    print("Computes allow list is not part of model tags")


## Create Infraestructure
# If you have a specific compute size to work with change it here. By default we use the 1 x V100 compute from the above list
compute_cluster_size = "Standard_NC6s_v3"

# If you already have a gpu cluster, mention it here. Else will create a new one with the name 'gpu-cluster-big'
compute_cluster = "gpu-cluster-big"

compute, gpus_per_node = create_gpu_cluster(
    workspace_ml_client, compute_cluster, compute_cluster_size, computes_allow_list
)


DATA_DIR = os.path.join(ROOT_DIR, "emotion-dataset")
# load the id2label json element of the ./emotion-dataset/label.json file into pandas table with keys as 'label' column of int64 type and values as 'label_string' column as string type

label_json_path = os.path.join(ROOT_DIR, "emotion-dataset", "label.json")

label_df = get_labels_dataset(label_json_path)
# check if Dataframe was loaded correctly
if len(label_df) == 0:
    print(f"Label Dataframe is empty")
    sys.exit(1)

print(f"Label Dataframe size {len(label_df)}")


train_path = os.path.join(ROOT_DIR, "emotion-dataset", "train.jsonl")
test_path = os.path.join(ROOT_DIR, "emotion-dataset", "test.jsonl")
val_path = os.path.join(ROOT_DIR, "emotion-dataset", "validation.jsonl")


# Create train, test , validation datasets
test_df, train_df, validation_df = create_datasets(
    data_path=DATA_DIR,
    train_path=train_path,
    test_path=test_path,
    val_path=val_path,
    label_dataset=label_df,
)


# check if Dataframe was loaded correctly
if len(train_df) == 0:
    print(f"Train Dataframe is empty")
    sys.exit(1)

print(f"Train Dataframe size {len(train_df)}")

if len(test_df) == 0:
    print(f"Test Dataframe is empty")
    sys.exit(1)

print(f"Test Dataframe size {len(test_df)}")

if len(validation_df) == 0:
    print(f"Validation Dataframe is empty")
    sys.exit(1)

print(f"Validation Dataframe size {len(validation_df)}")


# Training parameters
training_parameters = dict(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    metric_for_best_model="f1_macro",
)
print(f"The following training parameters are enabled - {training_parameters}")

# Optimization parameters - As these parameters are packaged with the model itself, lets retrieve those parameters
if "model_specific_defaults" in foundation_model.tags:
    optimization_parameters = ast.literal_eval(
        foundation_model.tags["model_specific_defaults"]
    )  # convert string to python dict
else:
    optimization_parameters = dict(
        apply_lora="true", apply_deepspeed="true", apply_ort="true"
    )
print(f"The following optimizations are enabled - {optimization_parameters}")


# fetch the pipeline component
pipeline_component_func = registry_ml_client.components.get(
    name="text_classification_pipeline", label="latest"
)


# define the pipeline job
# @pipeline(compute="serverless") # if you plan to use serverless uncomment this line and comment the next line
@pipeline()
def create_pipeline():
    text_classification_pipeline = pipeline_component_func(
        # specify the foundation model available in the azureml system registry id identified in step #3
        mlflow_model_path=foundation_model.id,
        # huggingface_id = 'bert-base-uncased', # if you want to use a huggingface model, uncomment this line and comment the above line
        compute_model_import=compute_cluster,
        compute_preprocess=compute_cluster,
        compute_finetune=compute_cluster,
        compute_model_evaluation=compute_cluster,
        # map the dataset splits to parameters
        train_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_train.jsonl"
        ),
        validation_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_validation.jsonl"
        ),
        test_file_path=Input(
            type="uri_file", path="./emotion-dataset/small_test.jsonl"
        ),
        evaluation_config=Input(
            type="uri_file", path="./text-classification-config.json"
        ),
        # The following parameters map to the dataset fields
        sentence1_key="text",
        label_key="label_string",
        # Training settings
        number_of_gpu_to_use_finetuning=gpus_per_node,  # set to the number of GPUs available in the compute
        **training_parameters,
        **optimization_parameters,
    )
    return {
        # map the output of the fine tuning job to the output of pipeline job so that we can easily register the fine tuned model
        # registering the model is required to deploy the model to an online or batch endpoint
        "trained_model": text_classification_pipeline.outputs.mlflow_model_folder
    }


pipeline_object = create_pipeline()

# don't use cached results from previous jobs
pipeline_object.settings.force_rerun = False

# set continue on step failure to False
pipeline_object.settings.continue_on_step_failure = False


# submit the pipeline job
pipeline_job = workspace_ml_client.jobs.create_or_update(
    pipeline_object, experiment_name=experiment_name
)
# wait for the pipeline job to complete
workspace_ml_client.jobs.stream(pipeline_job.name)


# Review training and evaluation metrics

# refresh the latest status of the job after streaming
returned_job = workspace_ml_client.jobs.get(name=pipeline_job.name)
if returned_job.status == "Completed":
    print("Job completed")

mlflow_tracking_uri = workspace_ml_client.workspaces.get(
    workspace_ml_client.workspace_name
).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
# concat 'tags.mlflow.rootRunId=' and pipeline_job.name in single quotes as filter variable
filter = "tags.mlflow.rootRunId='" + pipeline_job.name + "'"


runs = mlflow.search_runs(
    experiment_names=[experiment_name], filter_string=filter, output_format="list"
)
training_run = None
evaluation_run = None
# get the training and evaluation runs.
# using a hacky way till 'Bug 2320997: not able to show eval metrics in FT notebooks - mlflow client now showing display names' is fixed
for run in runs:
    # check if run.data.metrics.epoch exists
    if "epoch" in run.data.metrics:
        training_run = run
    # else, check if run.data.metrics.accuracy exists
    elif "accuracy" in run.data.metrics:
        evaluation_run = run


if training_run:
    print("Training metrics:\n\n")
    print(json.dumps(training_run.data.metrics, indent=2))
else:
    print("No Training job found")

if evaluation_run:
    print("Evaluation metrics:\n\n")
    print(json.dumps(evaluation_run.data.metrics, indent=2))
else:
    print("No Evaluation job found")

print("Registring Model ...... ")
# Register the fine tuned model with the workspace

# check if the `trained_model` output is available
print("pipeline job outputs: ", workspace_ml_client.jobs.get(pipeline_job.name).outputs)

# fetch the model from pipeline job output - not working, hence fetching from fine tune child job
model_path_from_job = "azureml://jobs/{0}/outputs/{1}".format(
    pipeline_job.name, "trained_model"
)

finetuned_model_name = model_name + "-emotion-detection"
finetuned_model_name = finetuned_model_name.replace("/", "-")
print("path to register model: ", model_path_from_job)
prepare_to_register_model = Model(
    path=model_path_from_job,
    type=AssetTypes.MLFLOW_MODEL,
    name=finetuned_model_name,
    version=timestamp,  # use timestamp as version to avoid version conflict
    description=model_name + " fine tuned model for emotion detection",
)
print("prepare to register model: \n", prepare_to_register_model)
# register the model from pipeline job output
registered_model = workspace_ml_client.models.create_or_update(
    prepare_to_register_model
)
print("registered model: \n", registered_model)


# Deploy the fine tuned model to an online endpoint
print("Deploint Model to Online Endpoint...... ")
# Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name

online_endpoint_name = "emotion-" + timestamp
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for "
    + registered_model.name
    + ", fine tuned model for emotion detection",
    auth_mode="key",
)
# create the endpoint and wait up to it is done
workspace_ml_client.begin_create_or_update(endpoint).wait()

# create a deployment
demo_deployment = ManagedOnlineDeployment(
    name="demo",
    endpoint_name=online_endpoint_name,
    model=registered_model.id,
    instance_type="Standard_DS3_v2",  # "Standard_E64s_v3", if llama2 the  instance type shall be bigger
    instance_count=1,
    liveness_probe=ProbeSettings(initial_delay=600),
)
workspace_ml_client.online_deployments.begin_create_or_update(demo_deployment).wait()
endpoint.traffic = {"demo": 100}
workspace_ml_client.begin_create_or_update(endpoint).result()
