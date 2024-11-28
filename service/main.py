from fastapi.responses import StreamingResponse
from serialization import load_model
from fastapi import FastAPI, File, UploadFile
from typing import List
from model import Item
import pandas as pd
import io

app = FastAPI()
model = load_model('../model.pickle')


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.DataFrame.from_records([item.model_dump()])
    return model.predict(data)


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> StreamingResponse:
    data = pd.read_csv(file.file)
    data['selling_price'] = model.predict(data)
    stream = io.BytesIO()
    data.to_csv(stream)
    stream.seek(0)
    return StreamingResponse(
        stream, media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{file.filename}"'},
    )
