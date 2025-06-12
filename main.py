from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import sklearn

app_dir = Path(__file__).parent



app = FastAPI()

total_queries = 0
successful_queries = 0

try:
    with open(app_dir/'model_with_preprocessor.pkl', 'rb') as f:
        model = pickle.load(f)

    model_loaded = True
    model_name = 'random_forest'
except Exception as e:
    model_loaded = False
    model_name = "none"
    print(f"Error loading model: {e}")



class InferenceInput(BaseModel):
    Promo_start: str
    Promo_end: str
    Shipping_start: str
    Shipping_end: str
    Promo_type: str
    Feat_2: float
    Feat_3: float
    Agent: str
    Promo_id: str
    Item_id: str
    Feat_7: float
    Promo_class: str
    Feat_9: float
    Feat_10: float
    Feat_11: float
    Feat_12: int

@app.get("/")
async def root():
    return {"message": "ML Model Server is running!"}


@app.get("/ping")
async def ping() -> Dict[str, Any]:
    global total_queries, successful_queries
    total_queries += 1
    successful_queries += 1

    status = "ok" if model_loaded else "model not loaded"

    return {
        "status": status,
        "total_queries": total_queries,
        "successful_queries": successful_queries
    }


@app.post("/inference")
async def inference(input_data: InferenceInput) -> Dict[str, Any]:
    global total_queries, successful_queries
    total_queries += 1

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:

        input_dict = input_data.model_dump()

        p = pd.DataFrame([input_dict])

        p['Promo_time'] = (pd.to_datetime(p['Promo_end']) - pd.to_datetime(p['Promo_start'])).dt.days
        p['Shipping_time'] = (pd.to_datetime(p['Shipping_end']) - pd.to_datetime(p['Shipping_start'])).dt.days
        print(p)
        print(model)

        prediction = model.predict(p)[0]

        successful_queries += 1

        return {
            "prediction": float(prediction),
            "model": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)