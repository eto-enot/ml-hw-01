from serialization import load_model
from fastapi import FastAPI
from typing import List
from model import Item
import pandas as pd

app = FastAPI()

model = load_model('model.dat')

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame.from_records([item.model_dump()])
    return model.predict(data)


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    data = pd.DataFrame.from_records([x.model_dump() for x in items])
    return model.predict(data)