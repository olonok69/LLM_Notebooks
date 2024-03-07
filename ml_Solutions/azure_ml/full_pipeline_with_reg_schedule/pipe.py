### Requeriments

# pip install mldesigner --user -q
# pip install azure-ai-ml azure-identity  python-dotenv

# https://github.com/kubeflow/pipelines
# Pipeline
# https://learn.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines?view=azureml-api-2
# https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.dsl?view=azure-python
# Component
# https://learn.microsoft.com/en-us/azure/machine-learning/concept-component?view=azureml-api-2
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2
# https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.commandcomponent?view=azure-python
# Debug
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-debug-pipeline-reuse-issues?view=azureml-api-2

# Datasets
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-explore-data?view=azureml-api-2

# Schedule Pipelines
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-schedule-pipeline-job?view=azureml-api-2&tabs=python
# Batch Pipeline
# https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk?view=azureml-api-2
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-pipeline-from-job?view=azureml-api-2&source=recommendations&tabs=python

# import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command
from azure.ai.ml import Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from dotenv import dotenv_values
from dotenv import load_dotenv
from src.login import get_ws_client
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment
import os
from azure.ai.ml.entities import (
    BatchEndpoint,
    PipelineComponentBatchDeployment,
    Data,
    Environment,
    Data,
)


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


ml_client = get_ws_client(credential, subscription_id, resource_group, workspace)

print(ml_client)


# Create Dataset and get it to train the model

my_path = "./data/default_of_credit_card_clients.csv"
# set the version number of the data asset
v1 = "initial"

my_data = Data(
    name="credit-card-dataset",
    version=v1,
    description="Credit card data",
    path=my_path,
    type=AssetTypes.URI_FILE,
)

## create data asset if it doesn't already exist:
try:
    data_asset = ml_client.data.get(name="credit-card-dataset", version=v1)
    print(
        f"Data asset already exists. Name: {my_data.name}, version: {my_data.version}"
    )
except:
    ml_client.data.create_or_update(my_data)
    print(f"Data asset created. Name: {my_data.name}, version: {my_data.version}")


# get a handle of the data asset and print the URI
credit_data = ml_client.data.get(name="credit-card-dataset", version="initial")
print(f"Data asset URI: {credit_data.path}")

#### CREATE ENVIRONMENT FOR PIPELINE FROM YAML FILE

dependencies_dir = "./dependencies"
custom_env_name = "aml-scikit-learn"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for Credit Card Defaults pipeline",
    tags={"scikit-learn": "0.24.2"},
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.3.0",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)


#### CREATE DATA PREP COMPONENT


data_prep_src_dir = "./components/data_prep"

data_prep_component = command(
    name="data_prep_credit_defaults",
    display_name="Data preparation for training",
    description="reads a .xl input, split the input to train and test",
    inputs={
        "data": Input(type="uri_folder"),
        "test_train_ratio": Input(type="number"),
    },
    outputs=dict(
        train_data=Output(type="uri_folder", mode="rw_mount"),
        test_data=Output(type="uri_folder", mode="rw_mount"),
    ),
    # The source folder of the component
    code=data_prep_src_dir,
    command="""python data_prep.py \
            --data ${{inputs.data}} --test_train_ratio ${{inputs.test_train_ratio}} \
            --train_data ${{outputs.train_data}} --test_data ${{outputs.test_data}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
data_prep_component = ml_client.create_or_update(data_prep_component.component)

# Create (register) the component in your workspace
print(
    f"Component {data_prep_component.name} with Version {data_prep_component.version} is registered"
)

##### CREATE TRAIN COMPONENT

train_src_dir = "./components/train"
# importing the Component Package
from azure.ai.ml import load_component

# Loading the component from the yml file
train_component = load_component(source=os.path.join(train_src_dir, "train.yml"))

# Now we register the component to the workspace
train_component = ml_client.create_or_update(train_component)

# Create (register) the component in your workspace
print(
    f"Component {train_component.name} with Version {train_component.version} is registered"
)

#### CREATE PIPELINE

# the dsl decorator tells the sdk that we are defining an Azure Machine Learning pipeline
from azure.ai.ml import dsl, Input, Output


@dsl.pipeline(
    compute="serverless",  # "serverless" value runs pipeline on serverless compute
    description="E2E data_perp-train pipeline",
    force_rerun=True,
)
def credit_defaults_pipeline(
    pipeline_job_data_input,
    pipeline_job_test_train_ratio,
    pipeline_job_learning_rate,
    pipeline_job_registered_model_name,
):
    # using data_prep_function like a python call with its own inputs
    data_prep_job = data_prep_component(
        data=pipeline_job_data_input,
        test_train_ratio=pipeline_job_test_train_ratio,
    )

    # using train_func like a python call with its own inputs
    train_job = train_component(
        train_data=data_prep_job.outputs.train_data,  # note: using outputs from previous step
        test_data=data_prep_job.outputs.test_data,  # note: using outputs from previous step
        learning_rate=pipeline_job_learning_rate,  # note: using a pipeline input as parameter
        registered_model_name=pipeline_job_registered_model_name,
    )

    # a pipeline returns a dictionary of outputs
    # keys will code for the pipeline output identifier
    return {
        "pipeline_job_train_data": data_prep_job.outputs.train_data,
        "pipeline_job_test_data": data_prep_job.outputs.test_data,
    }


registered_model_name = "credit_defaults_model"

# Let's instantiate the pipeline with the parameters of our choice
pipeline = credit_defaults_pipeline(
    pipeline_job_data_input=Input(type="uri_file", path=credit_data.path),
    pipeline_job_test_train_ratio=0.25,
    pipeline_job_learning_rate=0.05,
    pipeline_job_registered_model_name=registered_model_name,
)

##### SUBMIT PIPELINE JOB
pipeline_job = ml_client.jobs.create_or_update(
    pipeline,
    # Project's name
    experiment_name="e2e_final_components",
)
ml_client.jobs.stream(pipeline_job.name)

# refresh the latest status of the job after streaming
returned_job = ml_client.jobs.get(name=pipeline_job.name)

if returned_job.status == "Completed":
    print("Pipeline job completed.")

### Create BATCH ENDPOINT
pipeline_component = pipeline_job.component
# Instantiate default values
pipeline_component.inputs["pipeline_job_data_input"] = Input(
    type="uri_file", path=credit_data.path
)
pipeline_component.inputs["pipeline_job_test_train_ratio"] = Input(
    type="number", default=0.25
)
pipeline_component.inputs["pipeline_job_learning_rate"] = Input(
    type="number", default=0.05
)
pipeline_component.inputs["pipeline_job_registered_model_name"] = Input(
    type="string", default=registered_model_name
)
# Register pipeline in components
ml_client.components.create_or_update(pipeline_component)

##### Define and Create BATCH endpoint
endpoint_name = "credit-default-batch-v2"
endpoint = BatchEndpoint(
    name=endpoint_name,
    description="Batch Endpoint for credit default model",
)

batch_endpoint = ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
print(batch_endpoint)

##### CREATE DEPLOYMENT
deployment_name = "credit-default-batch-deployment"
deployment = PipelineComponentBatchDeployment(
    name=deployment_name,
    description="deployment component credit default",
    endpoint_name=endpoint.name,
    component=pipeline_component,
    settings={"default_compute": "serverless", "continue_on_step_failure": False},
)

ml_client.batch_deployments.begin_create_or_update(deployment).result()


### TEST BATCH ENDPOINT

job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint_name,
    deployment_name=deployment_name,
    kwargs={"force_rerun": True},
    inputs={
        "pipeline_job_data_input": Input(type="uri_file", path=credit_data.path),
        "pipeline_job_learning_rate": Input(type="number", default=0.02),
    },
)


### Schedule OPTIONAL

"""
schedule_name = "simple_sdk_create_schedule_recurrence"

schedule_start_time = datetime.utcnow()
recurrence_trigger = RecurrenceTrigger(
    frequency="day",
    interval=1,
    schedule=RecurrencePattern(hours=10, minutes=[0, 1]),
    start_time=schedule_start_time,
    time_zone=TimeZone.UTC,
)

job_schedule = JobSchedule(
    name=schedule_name, trigger=recurrence_trigger, create_job=pipeline_job
)

"""
