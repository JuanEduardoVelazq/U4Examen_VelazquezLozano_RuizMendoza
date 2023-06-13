# Importamos las librerías necesarias
import uvicorn
from Diabetes import Diabetes  # Importamos la clase Diabetes desde el archivo Diabetes.py
from fastapi import FastAPI
import pickle

app = FastAPI()

# Cargamos el modelo SVM previamente entrenado desde el archivo "classifier.pkl"
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

@app.get("/")
def index():
    return {"message": "¡Hola! Bienvenido al modelo de predicción de diabetes."}

@app.post("/predict")
def predict_svm(data: Diabetes):
    data = data.dict()
    Pregnancies = data["Pregnancies"]
    Glucose = data["Glucose"]
    BloodPressure = data["BloodPressure"]
    SkinThickness = data["SkinThickness"]
    Insulin = data["Insulin"]
    BMI = data["BMI"]
    DiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
    Age = data["Age"]

    # Realizamos la predicción utilizando el modelo cargado
    prediction = classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    # Asignamos una etiqueta descriptiva según la predicción
    if prediction[0] == 1:
        prediction_label = "Tiene diabetes"
    else:
        prediction_label = "No tiene diabetes"

    return {"prediction": prediction_label}

if __name__ == "__main__":
    # Ejecutamos el servidor utilizando Uvicorn en el host y puerto especificados
    uvicorn.run(app, host="127.0.0.1", port=8000)
