from serialization import load_model
from fastapi import FastAPI, File, UploadFile
from typing import List
from model import Item
import pandas as pd

app = FastAPI()
model = load_model('../model.pickle')


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame.from_records([item.model_dump()])
    return model.predict(data)


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> List[float]:
    return model.predict(pd.read_csv(file.file))
