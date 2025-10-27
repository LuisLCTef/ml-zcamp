import pickle
from fastapi import FastAPI
from typing import Dict, Any

app = FastAPI(title="customer-scoring-service")

# Load the pre-downloaded pipeline
with open("pipeline_v1.bin", "rb") as f:
    pipeline = pickle.load(f)

def predict_single(customer: Dict[str, Any]):
    result = pipeline.predict_proba([customer])[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)
    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)