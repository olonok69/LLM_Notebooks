from dotenv import load_dotenv
import os
import pandas as pd

env_path = os.path.join("keys", ".env")
load_dotenv(env_path)
# env variables
MODEL_PATH = os.getenv("MODEL_PATH")
LOCAL_ENV = os.getenv("LOCAL_ENV")
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER")
EMBEDDINGS = os.getenv("EMBEDDINGS")
BFLOAT16 = int(os.getenv("BFLOAT16"))

# print(BFLOAT16)
# print(EMBEDDINGS)
# print(MODEL_NAME)
# print(LOCAL_ENV)
# print(MODEL_PATH)

# where I am
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# data folder
data_folder = os.path.join(ROOT_DIR, "data")
# os.makedirs(data_folder, exist_ok=True)
class_file = os.path.join(data_folder, "classes.csv")

df = pd.read_csv(class_file)

print(ROOT_DIR)
print(class_file)
print(df.head(2))
