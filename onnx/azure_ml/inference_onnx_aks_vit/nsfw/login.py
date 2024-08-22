from azure.ai.ml import MLClient


def get_ws_client(
    credential,
    subscription_id: str,
    resource_group: str,
    workspace: str,
):
    """
    login to ML Worksace and return MLClient object

    :param credential: credential object
    :param subscription_id: subscription id
    :param resource_group: resource group name
    :param workspace: workspace name
    :return: MLClient object
    :rtype: MLClient
    :raises: Exception, if credential is not correct or workspace information is not correct.

    """
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
    return ml_client
