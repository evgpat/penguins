from fastapi import FastAPI, UploadFile, BackgroundTasks
from contextlib import asynccontextmanager

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PATH = 'https://github.com/evgpat/datasets/raw/refs/heads/main/penguins.csv'
ml_models = {}

def penguins_pipeline():
    data = pd.read_csv(PATH)
    data = data.dropna()

    le = preprocessing.LabelEncoder()
    X = data[["bill_length_mm", "flipper_length_mm"]]
    le.fit(data["species"])
    y = le.transform(data["species"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = Pipeline(
        steps=[("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11))]
    )
    clf.set_params().fit(X_train, y_train)

    return clf, le


@asynccontextmanager
async def lifespan(app: FastAPI):
    clf, le = penguins_pipeline()

    ml_models["clf"] = clf
    ml_models["le"] = le

    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {
        "Name": "Penguins Prediction",
        "description": "This is a penguins prediction model based on the bill length and flipper length of the bird.",
    }

@app.get("/predict/")
async def predict(bill_length_mm: float = 0.0, flipper_length_mm: float = 0.0):
    param = {
                "bill_length_mm": bill_length_mm,
                "flipper_length_mm": flipper_length_mm
            }
    if bill_length_mm <=0.0 or flipper_length_mm <=0.0:
        return {
            "parameters": param,
            "error message": "Invalid input values",
        }
    else:
        result = ml_models["clf"].predict([[bill_length_mm, flipper_length_mm]])
        return {
            "parameters": param,
            "result": ml_models["le"].inverse_transform(result)[0],
        }
