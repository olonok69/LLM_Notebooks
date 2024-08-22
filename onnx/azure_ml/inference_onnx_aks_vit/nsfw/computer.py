from azure.ai.ml.entities import AmlCompute


def create_gpu_cluster(
    workspace_ml_client, compute_cluster, compute_cluster_size, computes_allow_list=None
):
    try:
        compute = workspace_ml_client.compute.get(compute_cluster)
        print("The compute cluster already exists! Reusing it for the current run")
    except Exception as ex:
        print(
            f"Looks like the compute cluster doesn't exist. Creating a new one with compute size {compute_cluster_size}!"
        )
        try:
            print("Attempt #1 - Trying to create a dedicated compute")
            compute = AmlCompute(
                name=compute_cluster,
                size=compute_cluster_size,
                tier="Dedicated",
                max_instances=2,  # For multi node training set this to an integer value more than 1
            )
            workspace_ml_client.compute.begin_create_or_update(compute).wait()
        except Exception as e:
            try:
                print(
                    "Attempt #2 - Trying to create a low priority compute. Since this is a low priority compute, the job could get pre-empted before completion."
                )
                compute = AmlCompute(
                    name=compute_cluster,
                    size=compute_cluster_size,
                    tier="LowPriority",
                    max_instances=2,  # For multi node training set this to an integer value more than 1
                )
                workspace_ml_client.compute.begin_create_or_update(compute).wait()
            except Exception as e:
                print(e)
                raise ValueError(
                    f"WARNING! Compute size {compute_cluster_size} not available in workspace"
                )

    # Sanity check on the created compute
    compute = workspace_ml_client.compute.get(compute_cluster)
    if compute.provisioning_state.lower() == "failed":
        raise ValueError(
            f"Provisioning failed, Compute '{compute_cluster}' is in failed state. "
            f"please try creating a different compute"
        )

    if computes_allow_list is not None:
        computes_allow_list_lower_case = [x.lower() for x in computes_allow_list]
        if compute.size.lower() not in computes_allow_list_lower_case:
            raise ValueError(
                f"VM size {compute.size} is not in the allow-listed computes for finetuning"
            )
    else:
        # Computes with K80 GPUs are not supported
        unsupported_gpu_vm_list = [
            "standard_nc6",
            "standard_nc12",
            "standard_nc24",
            "standard_nc24r",
        ]
        if compute.size.lower() in unsupported_gpu_vm_list:
            raise ValueError(
                f"VM size {compute.size} is currently not supported for finetuning"
            )

    # This is the number of GPUs in a single node of the selected 'vm_size' compute.
    # Setting this to less than the number of GPUs will result in underutilized GPUs, taking longer to train.
    # Setting this to more than the number of GPUs will result in an error.
    gpu_count_found = False
    workspace_compute_sku_list = workspace_ml_client.compute.list_sizes()
    available_sku_sizes = []
    for compute_sku in workspace_compute_sku_list:
        available_sku_sizes.append(compute_sku.name)
        if compute_sku.name.lower() == compute.size.lower():
            gpus_per_node = compute_sku.gpus
            gpu_count_found = True
    # if gpu_count_found not found, then print an error
    if gpu_count_found:
        print(f"Number of GPU's in compute {compute.size}: {gpus_per_node}")
    else:
        raise ValueError(
            f"Number of GPU's in compute {compute.size} not found. Available skus are: {available_sku_sizes}."
            f"This should not happen. Please check the selected compute cluster: {compute_cluster} and try again."
        )

    return compute, gpus_per_node
