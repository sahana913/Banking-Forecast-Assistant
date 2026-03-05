from fastapi import FastAPI
import pandas as pd
from app.forecast_engine import predict_default, forecast_liquidity
from app.ollama_agent import ask_ollama

app = FastAPI(title="Bank Forecast API")

@app.get("/predict_default")
def default_api():
    df = pd.read_csv("data/loans.csv")
    preds = predict_default(df)
    return preds.head(10).to_dict(orient="records")

@app.get("/forecast_liquidity/{branch_id}")
def liquidity_api(branch_id: str):
    result = forecast_liquidity(branch_id)
    return result.to_dict(orient="records")

@app.get("/explain/{topic}")
def explain(topic: str):
    prompt = f"Explain banking forecast context about {topic}"
    response = ask_ollama(prompt)
    return {"response": response}
