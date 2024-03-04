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


# Import required libraries
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from dotenv import dotenv_values
from dotenv import load_dotenv
from src.login import get_ws_client


# load variables from .env file to request token for Service Principal Account
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

# Retrieve an already attached Azure Machine Learning Compute.
# specify aml compute name.
cluster_name = "cpu-cluster"

try:
    ml_client.compute.get(cluster_name)
except Exception:
    print("Creating a new cpu compute target...")
    compute = AmlCompute(
        name=cluster_name, size="STANDARD_D2_V2", min_instances=0, max_instances=4
    )
    ml_client.compute.begin_create_or_update(compute).result()
print(ml_client.compute.get(cluster_name))


# Load Components from yaml file
parent_dir = "."
train_model = load_component(source=parent_dir + "/train_model.yml")
score_data = load_component(source=parent_dir + "/score_data.yml")
eval_model = load_component(source=parent_dir + "/eval_model.yml")


# Construct pipeline
@pipeline(allow_reuse=False, description="Test Pipeline")
def pipeline_with_components_from_yaml(
    training_input,
    test_input,
    training_max_epochs=20,
    training_learning_rate=1.8,
    learning_rate_schedule="time-based",
):
    """E2E dummy train-score-eval pipeline with components defined via yaml."""
    # Call component obj as function: apply given inputs & parameters to create a node in pipeline
    train_with_sample_data = train_model(
        training_data=training_input,
        max_epochs=training_max_epochs,
        learning_rate=training_learning_rate,
        learning_rate_schedule=learning_rate_schedule,
    )

    score_with_sample_data = score_data(
        model_input=train_with_sample_data.outputs.model_output, test_data=test_input
    )
    score_with_sample_data.outputs.score_output.mode = "upload"

    eval_with_sample_data = eval_model(
        scoring_result=score_with_sample_data.outputs.score_output
    )

    # Return: pipeline outputs
    return {
        "trained_model": train_with_sample_data.outputs.model_output,
        "scored_data": score_with_sample_data.outputs.score_output,
        "evaluation_report": eval_with_sample_data.outputs.eval_output,
    }


pipeline_job = pipeline_with_components_from_yaml(
    training_input=Input(type="uri_folder", path=parent_dir + "/data/"),
    test_input=Input(type="uri_folder", path=parent_dir + "/data/"),
    training_max_epochs=20,
    training_learning_rate=1.8,
    learning_rate_schedule="time-based",
)

# set pipeline level compute
pipeline_job.settings.default_compute = "cpu-cluster"  # "serverless"
pipeline_job.settings.force_rerun = True

# Submit pipeline job to workspace
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)

ml_client.jobs.stream(pipeline_job.name)

# refresh the latest status of the job after streaming
returned_job = ml_client.jobs.get(name=pipeline_job.name)

if returned_job.status == "Completed":
    print("Pipeline job completed.")
