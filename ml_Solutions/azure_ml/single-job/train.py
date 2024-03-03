##########################################################################
# https://pypi.org/project/python-dotenv/
# https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli
# https://learn.microsoft.com/en-gb/python/api/overview/azure/ai-ml-readme?view=azure-python

# pip install azure-ai-ml azure-identity  python-dotenv

# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb

# https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication?view=azureml-api-2&tabs=sdk#configure-a-service-principal

# import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input
from dotenv import dotenv_values
from dotenv import load_dotenv

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes


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


try:
    ml_client = MLClient.from_config(credential=credential)
except Exception as ex:
    # NOTE: Update following workspace information if not correctly configure before
    client_config = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace,
    }

    if client_config["subscription_id"].startswith("<"):
        print(
            "please update your <SUBSCRIPTION_ID> <RESOURCE_GROUP> <AML_WORKSPACE_NAME> in notebook cell"
        )
        raise ex
    else:  # write and reload from config file
        import json, os

        config_path = "./config/config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as fo:
            fo.write(json.dumps(client_config))
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace,
        )
print(ml_client)


# specify aml compute name.
cpu_compute_target = "cpu-cluster"

try:
    ml_client.compute.get(cpu_compute_target)
except Exception:
    print("Creating a new cpu compute target...")
    compute = AmlCompute(
        name=cpu_compute_target, size="STANDARD_D2_V2", min_instances=0, max_instances=4
    )
    ml_client.compute.begin_create_or_update(compute).result()


# define the command
command_job = command(
    code="./src",
    command="python main.py --iris-csv ${{inputs.iris_csv}} --learning-rate ${{inputs.learning_rate}} --boosting ${{inputs.boosting}}",
    environment="AzureML-lightgbm-3.2-ubuntu18.04-py37-cpu@latest",
    inputs={
        "iris_csv": Input(
            type="uri_file",
            path="https://azuremlexamples.blob.core.windows.net/datasets/iris.csv",
        ),
        "learning_rate": 0.9,
        "boosting": "gbdt",
    },
    compute="cpu-cluster",
)
# submit the command
submitted_job = ml_client.jobs.create_or_update(command_job)
# stream the output and wait until the job is finished
ml_client.jobs.stream(submitted_job.name)

# refresh the latest status of the job after streaming
returned_job = ml_client.jobs.get(name=submitted_job.name)

if returned_job.status == "Completed":
    run_model = Model(
        path="azureml://jobs/{}/outputs/artifacts/paths/model/".format(
            returned_job.name
        ),
        name="run-model-example",
        description="Model created from run.",
        type=AssetTypes.MLFLOW_MODEL,
    )

    ml_client.models.create_or_update(run_model)

else:

    print(
        "Training job status: {}. Please wait until it completes".format(
            returned_job.status
        )
    )
