### Requeriments

# ! pip install mldesigner --user -q
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


# import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from dotenv import dotenv_values
from dotenv import load_dotenv
from src.login import get_ws_client

# load component function from component python file
from prep.prep_component import prepare_data_component
from train.train_component import keras_train_component


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


# INFRA
# specify aml compute name. CPU
cpu_compute_target = "cpu-cluster"

try:
    ml_client.compute.get(cpu_compute_target)
except Exception:
    print("Creating a new cpu compute target...")
    compute = AmlCompute(
        name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
    )
    ml_client.compute.begin_create_or_update(compute).result()

# specify aml compute name. GPU
gpu_compute_target = "gpu-cluster"

try:
    ml_client.compute.get(gpu_compute_target)
except Exception:
    print("Creating a new gpu compute target...")
    compute = AmlCompute(
        name=gpu_compute_target,
        size="STANDARD_NC6s_v3",
        min_instances=0,
        max_instances=4,
    )
    ml_client.compute.begin_create_or_update(compute).result()

fashion_ds = Input(
    path="wasbs://demo@data4mldemo6150520719.blob.core.windows.net/mnist-fashion/"
)

# load component function from yaml
keras_score_component = load_component(source="./score/score.yaml")


# define a pipeline containing 3 nodes: Prepare data node, train node, and score node
@pipeline(default_compute=cpu_compute_target, force_rerun=False)
def image_classification_keras_minist_convnet(pipeline_input_data):
    """E2E image classification pipeline with keras using python sdk."""
    prepare_data_node = prepare_data_component(input_data=pipeline_input_data)

    train_node = keras_train_component(
        input_data=prepare_data_node.outputs.training_data
    )
    train_node.compute = gpu_compute_target

    score_node = keras_score_component(
        input_data=prepare_data_node.outputs.test_data,
        input_model=train_node.outputs.output_model,
    )


# create a pipeline
pipeline_job = image_classification_keras_minist_convnet(pipeline_input_data=fashion_ds)


# SUBMIT PIPELINE JOB
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)

# wait until the job completes
ml_client.jobs.stream(pipeline_job.name)

components_names = [
    "prep_data",
    "train_image_classification_keras",
    "score_image_classification_keras",
]
components = [prepare_data_component, keras_train_component, keras_score_component]
get_components = {}
for c, cn in zip(components, components_names):
    try:
        # try get back the component
        prep = ml_client.components.get(name=cn, version="1")
    except:
        # if not exists, register component using following code
        prep = ml_client.components.create_or_update(c)

    get_components[cn] = prep

# list all components registered in workspace
for c in ml_client.components.list():
    print(c)
