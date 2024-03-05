### Requeriments

# pip install mldesigner --user -q
# pip install azure-ai-ml azure-identity  python-dotenv

# https://huggingface.co/tasks/question-answering

#### Deploy online Endpoints

# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints?view=azureml-api-2&tabs=python

### AUTHENTICATE ENDPOINTS
# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-online-endpoint?view=azureml-api-2&tabs=python

## https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.managedonlineendpoint?view=azure-python
## https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.managedonlinedeployment?view=azure-python


# Import required libraries
from azure.identity import DefaultAzureCredential
import time, sys
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
from dotenv import dotenv_values
from dotenv import load_dotenv
from src.login import get_ws_client
from azure.ai.ml import MLClient

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


workspace_ml_client = get_ws_client(
    credential, subscription_id, resource_group, workspace
)

print(workspace_ml_client)


# Connect to the HuggingFaceHub registry
registry_ml_client = MLClient(credential, registry_name="HuggingFace")
print(registry_ml_client)


model_name = "deepset-roberta-base-squad2"
foundation_model = registry_ml_client.models.get(model_name, version="18")
print(
    "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
        foundation_model.name, foundation_model.version, foundation_model.id
    )
)


# Create online endpoint - endpoint names need to be unique in a region, hence using timestamp to create unique endpoint name
timestamp = int(time.time())
online_endpoint_name = "question-answering-" + str(timestamp)
# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=online_endpoint_name,
    description="Online endpoint for "
    + foundation_model.name
    + ", for question-answering task",
    auth_mode="key",
)
workspace_ml_client.begin_create_or_update(endpoint).wait()


# create a deployment
qa_huggingface_deployment = ManagedOnlineDeployment(
    name="test-qa",
    endpoint_name=online_endpoint_name,
    model=foundation_model.id,
    instance_type="Standard_DS3_v2",
    instance_count=1,
)
workspace_ml_client.online_deployments.begin_create_or_update(
    qa_huggingface_deployment
).wait()
# online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for demo deployment
endpoint.traffic = {"test-qa": 100}
deployment_creation = workspace_ml_client.begin_create_or_update(endpoint).result()

print("End Deployment")
