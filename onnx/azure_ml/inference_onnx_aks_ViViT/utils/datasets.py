import json
import pandas as pd
import os


def get_labels_dataset(path):
    """
    #create label dataset  from path
    """
    label_df = pd.DataFrame(data=[], columns=["label", "label_string"])
    with open(path) as f:
        id2label = json.load(f)
        id2label = id2label["id2label"]
        label_df = pd.DataFrame.from_dict(
            id2label, orient="index", columns=["label_string"]
        )
        label_df["label"] = label_df.index.astype("int64")
        label_df = label_df[["label", "label_string"]]
    return label_df


def create_datasets(
    data_path: str,
    train_path: str,
    test_path: str,
    val_path: str,
    label_dataset: pd.DataFrame,
    frac: int = 1,
):
    # load test.jsonl, train.jsonl and validation.jsonl form the ./emotion-dataset folder into pandas dataframes
    test_df = pd.read_json(test_path, lines=True)
    train_df = pd.read_json(train_path, lines=True)
    validation_df = pd.read_json(val_path, lines=True)
    # join the train, validation and test dataframes with the id2label dataframe to get the label_string column
    train_df = train_df.merge(label_dataset, on="label", how="left")
    validation_df = validation_df.merge(label_dataset, on="label", how="left")
    test_df = test_df.merge(label_dataset, on="label", how="left")

    # save 10% of the rows from the train, validation and test dataframes into files with small_ prefix in the ./emotion-dataset folder

    train_df.sample(frac=frac).to_json(
        os.path.join(data_path, "small_train.jsonl"), orient="records", lines=True
    )
    validation_df.sample(frac=frac).to_json(
        os.path.join(data_path, "small_validation.jsonl"), orient="records", lines=True
    )
    test_df.sample(frac=frac).to_json(
        os.path.join(data_path, "small_test.jsonl"), orient="records", lines=True
    )
    return test_df, train_df, validation_df
