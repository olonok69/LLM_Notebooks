# Custom Classifier

## Description

FastAPI REST Application with Endpoints to Train and Predict custom document classification models and Manage Mlflow artifacts from user side


## Training endpoint

Train a model based in a set of documents and Labels provided by user. User also provide a Name for the experiment

#### Entrypoint

POST

https://Cambiame/custom_classifier/process

*payload*

- labels: classes or labels to train the model
- path: folder root containing the documents organized in folders according to the labels 
- experiment_name: Name  of the experiments in MLFlow

```json
{
    "labels": ["business",
    "entertainment",
  "food",
  "graphics",
  "historical",
  "medical",
  "politics",
  "space",
  "sport",
  "technologie"],
 "path": "dataset1",
 "experiment_name": "Custom_Classifier10"}
```
*Output*

```json
{
    "status": {
        "code": 200,
        "message": "Success"
    },
    "data": {
        "info": {
            "artifact_uri": "file:///home/customclassifier/mlflow/mlruns/111/959ab312cfb64bc6abfba40b2757e072/artifacts",
            "end_time": null,
            "experiment_id": "111",
            "lifecycle_stage": "active",
            "run_id": "959ab312cfb64bc6abfba40b2757e072",
            "run_name": "training_classifier2024-07-18_091001",
            "run_uuid": "959ab312cfb64bc6abfba40b2757e072",
            "start_time": 1721293801094,
            "status": "RUNNING",
            "user_id": "root"
        },
        "data": {
            "metrics": {},
            "params": {},
            "tags": {
                "mlflow.user": "root",
                "mlflow.source.name": "/usr/local/bin/uvicorn",
                "mlflow.source.type": "LOCAL",
                "mlflow.runName": "training_classifier2024-07-18_091001"
            }
        }
    },
    "error": "",
    "number_documents_treated": 0,
    "number_documents_non_treated": 0,
    "list_id_not_treated": []
}
```

## Inference endpoint

Classify a document using a fined tuned model

#### Entrypoint

POST

https://Cambiame/custom_classifier/predict

*payload*

- runid: Mlflow experiment runID (returned to the UI after training)
- text: Text to classify

```json
{"runid": "2092651bbf294a918310ad609aab2a09", 
"text": "text to classify"}
```

*Output*


- label: Predicted Label
- Prediction : Predicted class 
- score : probability returned


```json
{"status": {"code": 200, "message": "Success"}, 
"data": {"label": "non-cv", "prediction": 0, "score": 0.9999998807907104}, 
"error": "", 
"number_documents_treated": 0, 
"number_documents_non_treated": 0, 
"list_id_not_treated": []}
```

## Jobs Status Experiment

Get the status of a Experiment. Last 3 runs

#### Entrypoint

POST

https://Cambiame/custom_classifier/job_status

*payload*

- experiment_name: Mlflow Experiment Name


```json
{
    "experiment_name": "Custom_Classifier"
}
```

*Output*

Last 3 runids of experiment_name

- run_id: RunId
- status : status of that RunID
- params.label2id : Direct Labels Dictionary
- params.id2label : Reverse Labels Dictionary
- tags.mlflow.runName : mlflow RunName


```json
[
    {
        "run_id": "959ab312cfb64bc6abfba40b2757e072",
        "status": "RUNNING",
        "params.label2id": "{'non-cv': 0, 'cv': 1}",
        "params.id2label": "{0: 'non-cv', 1: 'cv'}",
        "tags.mlflow.runName": "training_classifier2024-07-18_091001"
    },
    {
        "run_id": "f60c6fd37c1e4e05a62b6f6589cff0f0",
        "status": "FINISHED",
        "params.label2id": null,
        "params.id2label": null,
        "tags.mlflow.runName": "dataset_info 2024-07-18_090959"
    },
    {
        "run_id": "a7f64d4932454f64a8bc220dda780239",
        "status": "FINISHED",
        "params.label2id": "{'non-cv': 0, 'cv': 1}",
        "params.id2label": "{0: 'non-cv', 1: 'cv'}",
        "tags.mlflow.runName": "training_classifier2024-07-18_090225"
    }
]
```


## Experiment Soft Delete

Delete Experiment from UI (Not from Database)

#### Entrypoint

POST

https://Cambiame/custom_classifier/experiment/soft_delete

*payload*

- experiment_name: Mlflow Experiment Name


```json
{
    "experiment_name": "Custom_Classifier"
}
```

*Output*


- status : status of deleting that experiment

```json
{
    "status": "Problem deleting Experiment Custom_Classifier2. Status deleted"
}
```

## Experiment Hard Delete

Delete Experiment from Database and artifacts from Artifacts Repository

#### Entrypoint

POST

https://Cambiame/custom_classifier/experiment/hard_delete

*payload*

- experiment_name: Mlflow Experiment Name


```json
{
    "experiment_name": "Custom_Classifier"
}
```

*Output*


- status : status of deleting that experiment

```json
{
    "status": "Problem deleting Experiment Custom_Classifier2. Status deleted"
}
```

## RunID  Soft Delete

Delete RunID from UI Not from Database

#### Entrypoint

POST

https://Cambiame/custom_classifier/run/soft_delete

*payload*

- run_id: Mlflow RunId of Experiment


```json
{
    "run_id": "1fca3d03b0e847eb859ebc49235639dd"
}
```

*Output*


- status : status of deleting that experiment

```json
{
    "status": "Problem deleting run 1fca3d03b0e847eb859ebc49235639dd. Status deleted"
}
```


## Jobs Status Experiments

Get last 10 runs

#### Entrypoint

GET

https://Cambiame/custom_classifier/all_experiments/last_run

*payload*

- No Payload



*Output*

Status of the last 10 RunIds 

- run_id: RunId
- status : status of that RunID
- experiment_name : Experiment Name
- tags.mlflow.runName : mlflow RunName


```json
[
    {
        "run_id": "8822b18328664575b8c7cfb300bb1a04",
        "status": "FINISHED",
        "experiment_name": "uga24",
        "tags.mlflow.runName": "training_classifier2024-06-26_155526"
    },
    {
        "run_id": "52f9996dcfd34c76a34bb8266242a790",
        "status": "FINISHED",
        "experiment_name": "Curated CV Model",
        "tags.mlflow.runName": "training_classifier2024-06-26_003807"
    },
    {
        "run_id": "0109ccad436146bd82acaeebdc0ce4d9",
        "status": "FINISHED",
        "experiment_name": "cv test experiment - 20-06",
        "tags.mlflow.runName": "training_classifier2024-06-20_153753"
    },
    {
        "run_id": "db7e0bd7e6964dd69a53d9b4571a2db0",
        "status": "FINISHED",
        "experiment_name": "integration-test-34273",
        "tags.mlflow.runName": "training_classifier2024-06-20_114008"
    },
    {
        "run_id": "823a544c7ba24f86a1d864c476b95b54",
        "status": "FINISHED",
        "experiment_name": "integration-test-51617",
        "tags.mlflow.runName": "training_classifier2024-06-20_111613"
    },
    {
        "run_id": "ffd856b4fd6e4784bd513c189c7715a7",
        "status": "FINISHED",
        "experiment_name": "the cv test experiment",
        "tags.mlflow.runName": "training_classifier2024-06-20_104249"
    },
    {
        "run_id": "e1e588c68b314a4b8f7ab91f9504a177",
        "status": "FINISHED",
        "experiment_name": "integration-test-34980",
        "tags.mlflow.runName": "training_classifier2024-06-19_103711"
    },
    {
        "run_id": "f8c815d4da6f41aa8844286591cfd114",
        "status": "FINISHED",
        "experiment_name": "integration-test-98804",
        "tags.mlflow.runName": "training_classifier2024-06-19_095222"
    },
    {
        "run_id": "dda73541c34547faafce9a835dbd5c7b",
        "status": "FINISHED",
        "experiment_name": "integration-test-24255",
        "tags.mlflow.runName": "training_classifier2024-06-19_094158"
    },
    {
        "run_id": "55ba61f57a394ad9876da36e6c424326",
        "status": "FINISHED",
        "experiment_name": "integration-test-87318",
        "tags.mlflow.runName": "training_classifier2024-06-19_093055"
    },
    {
        "run_id": "f13de0308c1c41979e790ef37c2fa857",
        "status": "FINISHED",
        "experiment_name": "integration-test-81700",
        "tags.mlflow.runName": "training_classifier2024-06-19_092148"
    },
    {
        "run_id": "959ab312cfb64bc6abfba40b2757e072",
        "status": "FINISHED",
        "experiment_name": "Custom_Classifier",
        "tags.mlflow.runName": "training_classifier2024-07-18_091001"
    },
    {
        "run_id": "96f44253a48a451daea6cbd8c7cec398",
        "status": "FINISHED",
        "experiment_name": "integration-test-99671",
        "tags.mlflow.runName": "training_classifier2024-06-18_145635"
    },
    {
        "run_id": "32d60a2d26fa4cfb9d4e215ad743f2bc",
        "status": "FINISHED",
        "experiment_name": "integration-test-18692",
        "tags.mlflow.runName": "training_classifier2024-06-18_111028"
    },
    {
        "run_id": "7fb75492f2e8469c88b8a71df568fe71",
        "status": "FINISHED",
        "experiment_name": "integration-test",
        "tags.mlflow.runName": "training_classifier2024-06-18_104833"
    },
    {
        "run_id": "83f1e7e0ddac41df902afc216bbacb50",
        "status": "FINISHED",
        "experiment_name": "integration-test-1",
        "tags.mlflow.runName": "training_classifier2024-06-18_110527"
    },
    {
        "run_id": "0ac44d8ba69948a8b9dc83f71e0941d9",
        "status": "FINISHED",
        "experiment_name": "pm-4-demo-custom_classifier2",
        "tags.mlflow.runName": "training_classifier2024-06-14_160012"
    },
    {
        "run_id": "04f3143420664b7394c1fcbbcc7f481b",
        "status": "FINISHED",
        "experiment_name": "pm-3-jaz-custom_classifier2",
        "tags.mlflow.runName": "training_classifier2024-06-14_150604"
    },
    {
        "run_id": "2e2bbd1da1dc49c68e0b1d68558f0e60",
        "status": "FINISHED",
        "experiment_name": "pm-3-custom_classifier2",
        "tags.mlflow.runName": "training_classifier2024-06-14_150024"
    },
    {
        "run_id": "7b20a347a29f4d0f9cd1fd4ea98ca1c8",
        "status": "FINISHED",
        "experiment_name": "pm-2-custom_classifier2",
        "tags.mlflow.runName": "training_classifier2024-06-14_145543"
    },
    {
        "run_id": "b1cb72182dda4d42a235678342b6937d",
        "status": "FINISHED",
        "experiment_name": "pm-ibrar-custom_classifier2",
        "tags.mlflow.runName": "training_classifier2024-06-14_101648"
    },
    {
        "run_id": "244a9ce730d448baba71fb6103326529",
        "status": "FINISHED",
        "experiment_name": "Custom_Classifier10",
        "tags.mlflow.runName": "training_classifier2024-07-15_103247"
    }
]
```

## RunID Status

Get Status last 3 Runs of Experiment X

#### Entrypoint

GET

https://Cambiame/custom_classifier/runid/status?experiment_name=Custom_Classifier

*payload*

- No Payload

*Output*

Status of the last 3  RunIds 

- run_id: RunId
- status : status of that RunID
- params.label2id : Direct Labels Dictionary
- params.id2label : Reverse Labels Dictionary
- tags.mlflow.runName : mlflow RunName

```json
[
    {
        "run_id": "959ab312cfb64bc6abfba40b2757e072",
        "status": "FINISHED",
        "params.label2id": "{'non-cv': 0, 'cv': 1}",
        "params.id2label": "{0: 'non-cv', 1: 'cv'}",
        "tags.mlflow.runName": "training_classifier2024-07-18_091001"
    },
    {
        "run_id": "f60c6fd37c1e4e05a62b6f6589cff0f0",
        "status": "FINISHED",
        "params.label2id": null,
        "params.id2label": null,
        "tags.mlflow.runName": "dataset_info 2024-07-18_090959"
    },
    {
        "run_id": "a7f64d4932454f64a8bc220dda780239",
        "status": "FINISHED",
        "params.label2id": "{'non-cv': 0, 'cv': 1}",
        "params.id2label": "{0: 'non-cv', 1: 'cv'}",
        "tags.mlflow.runName": "training_classifier2024-07-18_090225"
    }
]
```
